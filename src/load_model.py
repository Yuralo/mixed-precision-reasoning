"""Model loading for full precision, CUDA bitsandbytes, and Mac fake quantization."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import resolve_device


@dataclass
class ModelLoadInfo:
    name_or_path: str
    quantization: str
    device: str
    dtype: str
    fake_quant_bits: int | None = None
    note: str | None = None


def _resolve_dtype(dtype: str, device: str) -> torch.dtype | str:
    if dtype == "auto":
        return torch.float16 if device in {"cuda", "mps"} else torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype {dtype!r}")
    return mapping[dtype]


@torch.no_grad()
def fake_quantize_model_(model: nn.Module, bits: int = 4) -> dict[str, int]:
    """Round Linear weights to signed per-tensor integers, then dequantize.

    This portable backend changes model numerics but not storage or speed. It is
    intentionally a debugging/research-signal backend, not a deployment claim.
    """
    if bits < 2 or bits > 16:
        raise ValueError("fake quantization bits must be between 2 and 16")
    qmax = (1 << (bits - 1)) - 1
    linears = 0
    parameters = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear) or module.weight is None:
            continue
        weight = module.weight.data
        max_abs = weight.abs().max()
        if not torch.isfinite(max_abs) or max_abs == 0:
            continue
        scale = max_abs / qmax
        weight.copy_(torch.clamp(torch.round(weight / scale), -qmax, qmax) * scale)
        linears += 1
        parameters += weight.numel()
    return {"linear_layers": linears, "quantized_parameters": parameters}


def load_model_and_tokenizer(
    name_or_path: str,
    quantization: str = "none",
    device: str = "auto",
    dtype: str = "auto",
    fake_quant_bits: int = 4,
    trust_remote_code: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    device = resolve_device(device)
    torch_dtype = _resolve_dtype(dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    note = None
    if quantization in {"bnb4", "bnb8"}:
        if device != "cuda":
            raise RuntimeError(f"{quantization} requires CUDA. Use --quantization fake on Mac/CPU.")
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantization == "bnb4",
            load_in_8bit=quantization == "bnb8",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantization not in {"none", "fake"}:
        raise ValueError("quantization must be one of: none, fake, bnb4, bnb8")

    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    fake_stats = None
    if quantization == "fake":
        fake_stats = fake_quantize_model_(model, fake_quant_bits)
        note = "Fake weight quantization changes numerics but does not reduce memory or latency."
    if quantization in {"none", "fake"}:
        model.to(device)
    model.eval()

    info = ModelLoadInfo(
        name_or_path=name_or_path,
        quantization=quantization,
        device=device,
        dtype=str(torch_dtype).replace("torch.", ""),
        fake_quant_bits=fake_quant_bits if quantization == "fake" else None,
        note=note,
    )
    payload = asdict(info)
    if fake_stats:
        payload["fake_quantization_stats"] = fake_stats
    return model, tokenizer, payload
