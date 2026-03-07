#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2026 Nobuo Tsukamoto
This software is released under the MIT License.
See the LICENSE file in the project root for more information.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


@dataclass
class SileroVADConfig:
    onnx_model_path: str
    sample_rate: int = 16000
    threshold: float = 0.5
    frame_len_sec: float = 0.032
    frame_overlap_sec: float = 0.0


class FrameVAD:
    def __init__(self, config: SileroVADConfig):
        model_path = Path(config.onnx_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"VAD model path does not exist: {model_path}")

        self.sample_rate = config.sample_rate
        self.threshold = config.threshold
        self.frame_len = config.frame_len_sec
        self.frame_overlap = config.frame_overlap_sec
        self.n_frame_len = int(self.frame_len * self.sample_rate)
        self.n_frame_overlap = int(self.frame_overlap * self.sample_rate)
        if self.n_frame_len <= 0:
            raise ValueError("frame_len_sec * sample_rate must be >= 1")

        # Keep the same buffering style as the original implementation.
        self.buffer = np.zeros(shape=2 * self.n_frame_overlap + self.n_frame_len, dtype=np.float32)
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]
        self._state = None
        self.reset()

    def reset(self):
        self.buffer.fill(0.0)
        self._state = None

    def _allocate_state(self) -> np.ndarray:
        # Silero ONNX commonly expects [2, batch, 128].
        return np.zeros((2, 1, 128), dtype=np.float32)

    def _build_inputs(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        # Use common input names in priority order to keep compatibility with Silero ONNX variants.
        inputs: dict[str, np.ndarray] = {}
        speech = frame.astype(np.float32)[None, :]
        sr = np.array(self.sample_rate, dtype=np.int64)
        state = self._allocate_state() if self._state is None else self._state

        for name in self.input_names:
            lname = name.lower()
            if lname in {"input", "audio", "x"}:
                inputs[name] = speech
            elif "sr" in lname or "rate" in lname:
                inputs[name] = sr
            elif "state" in lname or "hidden" in lname:
                inputs[name] = state
            else:
                # Fallback for unknown scalar/tensor inputs.
                if lname.endswith("context"):
                    inputs[name] = np.zeros((1, 64), dtype=np.float32)
                else:
                    inputs[name] = speech
        return inputs

    @staticmethod
    def _speech_probability(output: np.ndarray) -> float:
        arr = np.asarray(output)
        if arr.ndim == 0:
            return float(arr)
        flat = arr.reshape(-1)
        if flat.size == 1:
            return float(flat[0])
        if flat.size >= 2:
            # If logits of [non_speech, speech] are returned, use softmax for speech.
            logits = flat[:2]
            exp = np.exp(logits - np.max(logits))
            probs = exp / np.sum(exp)
            return float(probs[1])
        return 0.0

    def _decode(self, frame: np.ndarray):
        if len(frame) != self.n_frame_len:
            raise ValueError(f"Expected frame length {self.n_frame_len}, got {len(frame)}")

        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len :]
        self.buffer[-self.n_frame_len :] = frame
        inputs = self._build_inputs(self.buffer[-self.n_frame_len :])
        outputs = self.session.run(None, inputs)

        prob = self._speech_probability(outputs[0])

        # Save recurrent state if model returns it.
        for output in outputs[1:]:
            arr = np.asarray(output)
            if arr.ndim == 3 and arr.shape[0] == 2:
                self._state = arr.astype(np.float32)
                break

        pred = 1 if prob >= self.threshold else 0
        return [pred, "speech" if pred else "silence", 1.0 - prob, prob, str(prob)]

    def transcribe(self, frame: np.ndarray | None = None):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], "constant")
        elif len(frame) > self.n_frame_len:
            frame = frame[: self.n_frame_len]

        return self._decode(frame.astype(np.float32))
