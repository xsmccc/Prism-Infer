#include <cassert>

#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <45_dual_gemm/device/dual_gemm.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>

namespace {

template <
    typename ElementOutput_,
    int Count,
    typename ElementAccumulator_ = ElementOutput_,
    typename ElementCompute_ = float,
    cutlass::FloatRoundStyle Round =
        cutlass::FloatRoundStyle::round_to_nearest>
class EpilogueRoundedSiLuMul {
 public:
  static int const kCount = Count;
  static cutlass::FloatRoundStyle const kRound = Round;

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentCompute = cutlass::Array<ElementCompute, kCount>;

  struct Params {};

  CUTLASS_HOST_DEVICE
  EpilogueRoundedSiLuMul(Params const&) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return true;
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) {
    assert(false);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& gate,
      FragmentAccumulator const& up) const {
    cutlass::NumericArrayConverter<
        ElementCompute,
        ElementAccumulator,
        kCount,
        kRound>
        accumulator_to_compute;
    cutlass::NumericArrayConverter<
        ElementOutput,
        ElementCompute,
        kCount,
        kRound>
        compute_to_output;
    cutlass::NumericArrayConverter<
        ElementCompute,
        ElementOutput,
        kCount,
        kRound>
        output_to_compute;

    FragmentCompute gate_compute = accumulator_to_compute(gate);
    FragmentCompute up_compute = accumulator_to_compute(up);
    cutlass::epilogue::thread::SiLu<FragmentCompute> silu;
    FragmentOutput activated_bf16 = compute_to_output(silu(gate_compute));
    FragmentCompute activated_compute = output_to_compute(activated_bf16);
    cutlass::multiplies<FragmentCompute> multiply;
    return compute_to_output(multiply(activated_compute, up_compute));
  }
};

at::Tensor dual_gemm_swiglu_bf16(
    at::Tensor const& input,
    at::Tensor const& packed_weight) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::BFloat16,
      "input must be BF16");
  TORCH_CHECK(
      packed_weight.scalar_type() == at::ScalarType::BFloat16,
      "packed_weight must be BF16");
  TORCH_CHECK(input.dim() == 2, "input must be rank 2");
  TORCH_CHECK(packed_weight.dim() == 2, "packed_weight must be rank 2");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
  TORCH_CHECK(
      input.device() == packed_weight.device(),
      "input and packed_weight must be on the same device");
  TORCH_CHECK(
      packed_weight.size(0) % 2 == 0,
      "packed_weight rows must be divisible by 2");
  TORCH_CHECK(
      input.size(1) == packed_weight.size(1),
      "input and weight K dimensions must match");

  c10::cuda::CUDAGuard device_guard(input.device());

  int64_t const m = input.size(0);
  int64_t const k = input.size(1);
  int64_t const n = packed_weight.size(0) / 2;
  auto output = at::empty({m, n}, input.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using Element = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using EpilogueOutputOp01 = cutlass::epilogue::thread::LinearCombination<
      Element,
      128 / cutlass::sizeof_bits<Element>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using EpilogueOutputOp2 = EpilogueRoundedSiLuMul<
      Element,
      128 / cutlass::sizeof_bits<Element>::value,
      Element,
      ElementCompute>;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using DualGemm = cutlass::gemm::device::DualGemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::ColumnMajor,
      cutlass::layout::ColumnMajor,
      Element,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp01,
      EpilogueOutputOp01,
      EpilogueOutputOp2,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      3,
      false,
      false,
      false>;

  using RefA =
      cutlass::TensorRef<typename DualGemm::ElementA, typename DualGemm::LayoutA>;
  using RefB0 = cutlass::
      TensorRef<typename DualGemm::ElementB, typename DualGemm::LayoutB0>;
  using RefB1 = cutlass::
      TensorRef<typename DualGemm::ElementB, typename DualGemm::LayoutB1>;
  using RefC =
      cutlass::TensorRef<typename DualGemm::ElementC, typename DualGemm::LayoutC>;

  auto* input_ptr = reinterpret_cast<Element*>(input.data_ptr());
  auto* weight_ptr = reinterpret_cast<Element*>(packed_weight.data_ptr());
  auto* output_ptr = reinterpret_cast<Element*>(output.data_ptr());
  RefC empty_ref;

  typename DualGemm::Arguments arguments{
      cutlass::gemm::DualGemmMode::kGemm,
      cutlass::gemm::GemmCoord(m, n, k),
      RefA{input_ptr, typename DualGemm::LayoutA::Stride(input.stride(0))},
      RefB0{
          weight_ptr,
          typename DualGemm::LayoutB0::Stride(packed_weight.stride(0))},
      empty_ref,
      empty_ref,
      RefB1{
          weight_ptr + n * k,
          typename DualGemm::LayoutB1::Stride(packed_weight.stride(0))},
      empty_ref,
      empty_ref,
      RefC{
          output_ptr,
          typename DualGemm::LayoutC::Stride(output.stride(0))},
      typename DualGemm::EpilogueOutputOp0::Params{
          ElementCompute(1),
          ElementCompute(0)},
      typename DualGemm::EpilogueOutputOp1::Params{
          ElementCompute(1),
          ElementCompute(0)},
      typename DualGemm::EpilogueOutputOp2::Params{},
      1};

  DualGemm dual_gemm;
  cutlass::Status status = dual_gemm.can_implement(arguments);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "DualGemm cannot implement inputs: ",
      cutlass::cutlassGetStatusString(status));
  status = dual_gemm.initialize(arguments, nullptr, stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "DualGemm initialization failed: ",
      cutlass::cutlassGetStatusString(status));
  status = dual_gemm(stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "DualGemm launch failed: ",
      cutlass::cutlassGetStatusString(status));
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "dual_gemm_swiglu_bf16",
      &dual_gemm_swiglu_bf16,
      "BF16 dual-GEMM SwiGLU");
}
