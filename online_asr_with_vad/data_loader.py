#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType

import torch

import numpy as np


class AudioDataLayer(IterableDataset):
    """simple data layer to pass audio signal"""

    @property
    def output_types(self):
        return {
            "audio_signal": NeuralType(("B", "T"), AudioSignal(freq=self._sample_rate)),
            "a_sig_length": NeuralType(tuple("B"), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return (
            torch.as_tensor(self.signal, dtype=torch.float32),
            torch.as_tensor(self.signal_shape, dtype=torch.int64),
        )

    def set_signal(self, signal):
        self.signal = signal.astype(np.float32) / 32768.0
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1
