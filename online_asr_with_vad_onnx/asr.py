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
import tempfile

import numpy as np
import soundfile as sf

import onnxruntime_genai as og


@dataclass
class WhisperGenAIConfig:
    model_path: str
    execution_provider: str = "cpu"
    max_length: int = 448
    num_beams: int = 1
    prompt: str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"


class WhisperGenAIASR:
    def __init__(self, config: WhisperGenAIConfig):
        self._model_path = Path(config.model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"ASR model path does not exist: {self._model_path}")

        model_config = og.Config(str(self._model_path))
        model_config.clear_providers()

        provider = config.execution_provider.strip().lower()
        if provider == "cuda":
            model_config.append_provider("cuda")
        else:
            model_config.append_provider("cpu")

        self.model = og.Model(model_config)
        self.processor = self.model.create_multimodal_processor()
        self.max_length = config.max_length
        self.num_beams = config.num_beams
        self.prompt = config.prompt

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if audio.size == 0:
            return ""

        clipped = np.clip(audio, -1.0, 1.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, clipped, sample_rate, subtype="PCM_16")
            audios = og.Audios.open(tmp.name)
            inputs = self.processor([self.prompt], audios=audios)

            search_options = {
                "do_sample": False,
                "num_beams": self.num_beams,
                "num_return_sequences": 1,
                "max_length": self.max_length,
                "batch_size": 1,
            }

            params = og.GeneratorParams(self.model)
            params.set_search_options(**search_options)
            generator = og.Generator(self.model, params)
            generator.set_inputs(inputs)

            while not generator.is_done():
                generator.generate_next_token()

            sequence = generator.get_sequence(0)
            text = self.processor.decode(sequence)
            return text.strip()
