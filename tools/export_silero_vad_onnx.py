#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export Silero VAD (PyTorch) to ONNX.

This tool standardizes the ONNX signature to:
  inputs : input(float32[B,T]), sr(int64[1]), state(float32[2,B,128])
  outputs: output(float32[...]), state_out(float32[2,B,128])
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import onnxruntime as ort


class SileroExportWrapper:
    def __init__(self, torch_module: Any, model: Any, mode: str):
        self._torch = torch_module
        self.model = model
        self.mode = mode

    @staticmethod
    def _parse_output(result: Any, fallback_state: Any, torch_module: Any):
        if isinstance(result, (tuple, list)):
            if len(result) >= 2 and isinstance(result[1], torch_module.Tensor):
                return result[0], result[1]
            if len(result) >= 1:
                return result[0], fallback_state
        if isinstance(result, torch_module.Tensor):
            return result, fallback_state
        raise RuntimeError(f"Unexpected model output type: {type(result)}")

    def forward(self, x, sr, state):
        if self.mode == "x_sr":
            result = self.model(x, sr)
        elif self.mode == "x_state_sr":
            result = self.model(x, state, sr)
        elif self.mode == "x_sr_state":
            result = self.model(x, sr, state)
        else:
            raise RuntimeError(f"Unsupported forward mode: {self.mode}")

        out, next_state = self._parse_output(result, state, self._torch)
        return out, next_state


def detect_forward_mode(model: Any, sample_rate: int, frame_samples: int, torch_module: Any) -> str:
    x = torch_module.zeros(1, frame_samples, dtype=torch_module.float32)
    sr = torch_module.tensor([sample_rate], dtype=torch_module.int64)
    state = torch_module.zeros(2, 1, 128, dtype=torch_module.float32)

    candidates = [
        ("x_sr", (x, sr)),
        ("x_state_sr", (x, state, sr)),
        ("x_sr_state", (x, sr, state)),
    ]

    with torch_module.no_grad():
        for mode, args in candidates:
            try:
                _ = model(*args)
                return mode
            except Exception:
                continue

    raise RuntimeError("Failed to detect Silero VAD forward signature automatically.")


def export_silero_to_onnx(
    output_path: Path,
    sample_rate: int,
    opset: int,
    repo_or_dir: str,
    model_name: str,
    force_reload: bool,
    verify: bool,
    use_bundled_onnx: bool,
):
    import torch

    if sample_rate not in (8000, 16000):
        raise ValueError("sample_rate must be 8000 or 16000 for Silero VAD.")

    frame_samples = 256 if sample_rate == 8000 else 512
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, _utils = torch.hub.load(
        repo_or_dir=repo_or_dir,
        model=model_name,
        trust_repo=True,
        force_reload=force_reload,
    )
    model.eval()

    mode = detect_forward_mode(model, sample_rate=sample_rate, frame_samples=frame_samples, torch_module=torch)
    wrapped = SileroExportWrapper(torch_module=torch, model=model, mode=mode)

    x = torch.zeros(1, frame_samples, dtype=torch.float32)
    sr = torch.tensor([sample_rate], dtype=torch.int64)
    state = torch.zeros(2, 1, 128, dtype=torch.float32)

    class ExportModule(torch.nn.Module):
        def __init__(self, w: SileroExportWrapper):
            super().__init__()
            self.w = w

        def forward(self, x, sr, state):
            return self.w.forward(x, sr, state)

    export_module = ExportModule(wrapped).eval()

    exported = False
    try:
        torch.onnx.export(
            export_module,
            (x, sr, state),
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input", "sr", "state"],
            output_names=["output", "state_out"],
            dynamic_axes={
                "input": {0: "batch", 1: "samples"},
                "state": {1: "batch"},
                "output": {0: "batch"},
                "state_out": {1: "batch"},
            },
            dynamo=False,
        )
        exported = True
        print(f"[OK] Exported ONNX: {output_path}")
        print(f"[INFO] Detected forward signature mode: {mode}")
    except Exception as exc:
        if not use_bundled_onnx:
            raise
        # Fallback: copy ONNX bundled in Silero repo.
        hub_dir = Path(torch.hub.get_dir())
        repo_dir = hub_dir / "snakers4_silero-vad_master" / "src" / "silero_vad" / "data"
        candidate = repo_dir / "silero_vad_op18_ifless.onnx"
        if not candidate.exists():
            candidate = repo_dir / "silero_vad.onnx"
        if not candidate.exists():
            raise RuntimeError(f"ONNX export failed and no bundled ONNX found. export_error={exc}")
        shutil.copy2(candidate, output_path)
        exported = True
        print(f"[WARN] torch.onnx.export failed. Fallback to bundled ONNX: {candidate}")
        print(f"[OK] Copied ONNX: {output_path}")

    if verify and exported:
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        outs = session.run(
            None,
            {
                "input": x.numpy(),
                "sr": sr.numpy(),
                "state": state.numpy(),
            },
        )
        print(f"[OK] ONNX Runtime verification succeeded. outputs={len(outs)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export Silero VAD to ONNX")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("online_asr_with_vad_onnx/models/silero_vad.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        choices=[8000, 16000],
        help="Sampling rate for export dummy input",
    )
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    parser.add_argument(
        "--repo-or-dir",
        type=str,
        default="snakers4/silero-vad",
        help="torch.hub repo_or_dir",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="silero_vad",
        help="torch.hub model name",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force re-download from torch.hub",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX Runtime verification after export",
    )
    parser.add_argument(
        "--no-bundle-fallback",
        action="store_true",
        help="Disable fallback copy from Silero bundled ONNX when export fails",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_silero_to_onnx(
        output_path=args.output,
        sample_rate=args.sample_rate,
        opset=args.opset,
        repo_or_dir=args.repo_or_dir,
        model_name=args.model_name,
        force_reload=args.force_reload,
        verify=not args.no_verify,
        use_bundled_onnx=not args.no_bundle_fallback,
    )


if __name__ == "__main__":
    main()
