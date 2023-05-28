#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import copy

import numpy as np

import torch
from torch.utils.data import DataLoader

import nemo.collections.asr as nemo_asr

from data_loader import AudioDataLayer


class FrameVAD:
    def __init__(
        self,
        sample_rate=16000,
        threshold=0.5,
        frame_len=2,
        frame_overlap=2.5,
        offset=10,
    ):
        """
        Args:
          threshold: If prob of speech is larger than threshold, classify the segment to be speech.
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        """

        torch.device("cuda")
        # torch.device('cpu')

        vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
            "vad_marblenet"
        )
        cfg = copy.deepcopy(vad_model._cfg)
        vad_model.preprocessor = vad_model.from_config_dict(cfg.preprocessor)
        vad_model.eval()  # Set model to inference mode
        self.vad_model = vad_model.to(vad_model.device)

        self.vocab = cfg.labels
        self.vocab.append("_")

        self.sr = sample_rate
        self.threshold = threshold
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = cfg.preprocessor["window_stride"]
        for block in cfg.encoder["jasper"]:
            timestep_duration *= block["stride"][0] ** block["repeat"]
        self.buffer = np.zeros(
            shape=2 * self.n_frame_overlap + self.n_frame_len, dtype=np.float32
        )
        self.offset = offset

        self.data_layer = AudioDataLayer(sample_rate=cfg.train_ds.sample_rate)
        self.data_loader = DataLoader(
            self.data_layer, batch_size=1, collate_fn=self.data_layer.collate_fn
        )

        self.reset()

    def _decode(self, frame, offset=0):
        assert len(frame) == self.n_frame_len
        self.buffer[: -self.n_frame_len] = self.buffer[self.n_frame_len :]
        self.buffer[-self.n_frame_len :] = frame
        logits = self._infer_signal(self.buffer)
        decoded = self._greedy_decoder(self.threshold, logits, self.vocab)
        return decoded

    def _infer_signal(self, signal):
        """inference method for audio signal (single instance)"""
        self.data_layer.set_signal(signal)
        batch = next(iter(self.data_loader))
        audio_signal, audio_signal_len = batch
        audio_signal = audio_signal.to(self.vad_model.device)
        audio_signal_len = audio_signal_len.to(self.vad_model.device)
        processed_signal, processed_signal_len = self.vad_model.preprocessor(
            input_signal=audio_signal,
            length=audio_signal_len,
        )
        logits = self.vad_model.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )
        return logits.cpu().numpy()[0]

    @torch.no_grad()
    def transcribe(self, frame=None):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], "constant")
        unmerged = self._decode(frame, self.offset)
        return unmerged

    def reset(self):
        """
        Reset frame_history and decoder's state
        """
        self.buffer = np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ""

    @staticmethod
    def _greedy_decoder(threshold, logits, vocab):
        s = []
        if logits.shape[0]:
            probs = torch.softmax(torch.as_tensor(logits), dim=-1)
            probas, _ = torch.max(probs, dim=-1)
            probas_s = probs[1].item()
            preds = 1 if probas_s >= threshold else 0
            s = [
                preds,
                str(vocab[preds]),
                probs[0].item(),
                probs[1].item(),
                str(logits),
            ]
        return s
