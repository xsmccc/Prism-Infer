#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${repo_root}/data/p6_real_samples"

samples=(
  "000000039769|dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e"
  "000000037777|fe513b68ba46bc146114c469fdff52c288226073f03d73d3300d4fa739937369"
  "000000087038|c6cfc7e454a432c31ac3c3a997d5f33ccebe120fc6b9465965609ce12995ede9"
  "000000174482|41fab00aad97dad3eb7d035420b8cc61ed0ca8f4f0b66a4fbc1bd0bba6f8a5b3"
  "000000252219|1cf48bddb1bcab80d9f4d18514ae438c3e4f57d18e80c4fbea9d5d22396f514a"
  "000000397133|09e1d25c75f7879bdaa69c327fece5cabacd53939c8c2ef9e87f1c97a2e478c4"
  "000000403385|11632ed3fb470d62f7fe5f0445c4f10ec91225c4a820c95d1c3946af9426d4c7"
)

mkdir -p "${output_dir}"
rm -f "${output_dir}"/*.tmp

for sample in "${samples[@]}"; do
  IFS='|' read -r image_id expected_sha256 <<< "${sample}"
  output_path="${output_dir}/${image_id}.jpg"
  temporary_path="${output_path}.tmp"
  source_url="http://images.cocodataset.org/val2017/${image_id}.jpg"
  curl --fail --location --retry 3 --output "${temporary_path}" "${source_url}"
  printf '%s  %s\n' "${expected_sha256}" "${temporary_path}" \
    | sha256sum --check --status
done

for sample in "${samples[@]}"; do
  IFS='|' read -r image_id _ <<< "${sample}"
  output_path="${output_dir}/${image_id}.jpg"
  mv "${output_path}.tmp" "${output_path}"
  printf 'verified P6 real sample: %s\n' "${output_path}"
done
