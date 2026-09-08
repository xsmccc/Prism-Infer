"""Model construction must not leave a default-device dispatch mode installed."""

import pytest
import torch
from torch.overrides import _get_current_function_mode_stack

from prism_infer.engine.model_runner import _model_initialization_defaults


def test_model_initialization_removes_its_device_mode():
    original_modes = _get_current_function_mode_stack()
    original_dtype = torch.get_default_dtype()
    with _model_initialization_defaults(torch.bfloat16):
        assert torch.get_default_dtype() == torch.bfloat16
    assert _get_current_function_mode_stack() == original_modes
    assert torch.get_default_dtype() == original_dtype


def test_model_initialization_restores_caller_mode_after_exception():
    original_dtype = torch.get_default_dtype()
    with torch.device("cpu"):
        caller_modes = _get_current_function_mode_stack()
        with pytest.raises(RuntimeError, match="construction failed"):
            with _model_initialization_defaults(torch.bfloat16):
                raise RuntimeError("construction failed")
        assert _get_current_function_mode_stack() == caller_modes
        assert torch.empty(0).device.type == "cpu"
    assert torch.get_default_dtype() == original_dtype
