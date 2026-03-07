#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2026 Nobuo Tsukamoto
This software is released under the MIT License.
See the LICENSE file in the project root for more information.
"""

import argparse
import asyncio
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import yaml
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.text import Text

from asr import WhisperGenAIASR, WhisperGenAIConfig
from vad import FrameVAD, SileroVADConfig


@dataclass
class AudioConfig:
    input_device_id: int
    sample_rate: int
    channels: int
    frame_step_sec: float


@dataclass
class VADConfig:
    window_size_sec: float
    threshold: float
    check_history_frames: int
    min_history_for_decision_frames: int
    speech_ratio_threshold: float
    background_ratio_threshold: float
    onnx_model_path: str


@dataclass
class SegmentConfig:
    max_history_frames: int


@dataclass
class ASRConfig:
    model_path: str
    execution_provider: str
    max_length: int
    num_beams: int
    prompt: str


@dataclass
class LoggingConfig:
    level: str


@dataclass
class AppConfig:
    audio: AudioConfig
    vad: VADConfig
    segment: SegmentConfig
    asr: ASRConfig
    logging: LoggingConfig


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def build_status_text(vad_state: str, asr_state: str) -> Text:
    vad_width = 7
    asr_width = 8
    vad_label = vad_state.ljust(vad_width)
    asr_label = asr_state.ljust(asr_width)

    text = Text()
    text.append("VAD | ", style="bold white")
    text.append(vad_label, style="bold green" if vad_state == "Speech" else "bold yellow")
    text.append(" |  ", style="bold white")
    text.append("ASR | ", style="bold white")
    text.append(asr_label, style="bold cyan" if asr_state == "Decoding" else "bold bright_black")
    text.append(" |", style="bold white")
    return text


def configure_logger(config_level: str):
    log_level = os.getenv("LOG_LEVEL", config_level).upper()
    logger.remove()
    logger.add(sys.stderr, level=log_level)


async def inputstream_generator(channels: int, samplerate: int, chunk_size: int, device_id: int):
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(
        callback=callback,
        channels=channels,
        samplerate=samplerate,
        blocksize=chunk_size,
        dtype="int16",
        device=device_id,
    )
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


def load_config(config_path: Path) -> AppConfig:
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    audio = raw["audio"]
    vad = raw["vad"]
    segment = raw["segment"]
    asr = raw["asr"]
    logging_cfg = raw.get("logging", {})

    return AppConfig(
        audio=AudioConfig(
            input_device_id=int(audio["input_device_id"]),
            sample_rate=int(audio["sample_rate"]),
            channels=int(audio["channels"]),
            frame_step_sec=float(audio["frame_step_sec"]),
        ),
        vad=VADConfig(
            window_size_sec=float(vad["window_size_sec"]),
            threshold=float(vad["threshold"]),
            check_history_frames=int(vad["check_history_frames"]),
            min_history_for_decision_frames=int(vad["min_history_for_decision_frames"]),
            speech_ratio_threshold=float(vad["speech_ratio_threshold"]),
            background_ratio_threshold=float(vad["background_ratio_threshold"]),
            onnx_model_path=str(vad["onnx_model_path"]),
        ),
        segment=SegmentConfig(
            max_history_frames=int(segment["max_history_frames"]),
        ),
        asr=ASRConfig(
            model_path=str(asr["model_path"]),
            execution_provider=str(asr.get("execution_provider", "cpu")),
            max_length=int(asr.get("max_length", 448)),
            num_beams=int(asr.get("num_beams", 1)),
            prompt=str(asr.get("prompt", "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>")),
        ),
        logging=LoggingConfig(
            level=str(logging_cfg.get("level", "INFO")),
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Online ASR with Silero ONNX VAD and onnxruntime-genai Whisper")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser.parse_args()


async def mic_stream(config: AppConfig):
    window_size_sec = config.vad.window_size_sec
    frame_step_sec = config.audio.frame_step_sec
    threshold = config.vad.threshold
    samplerate = config.audio.sample_rate
    channels = config.audio.channels
    input_device_id = config.audio.input_device_id
    max_history_frames = config.segment.max_history_frames

    speech_count = 0
    is_speech = False
    speech_histories = []
    check_history = config.vad.check_history_frames
    min_history_for_decision = config.vad.min_history_for_decision_frames
    speech_ratio_threshold = config.vad.speech_ratio_threshold
    background_ratio_threshold = config.vad.background_ratio_threshold

    samples_per_frame = int(frame_step_sec * samplerate)
    frame_overlap_sec = (window_size_sec - frame_step_sec) / 2
    max_segment_sec = max_history_frames * frame_step_sec
    if samples_per_frame <= 0:
        raise ValueError("audio.frame_step_sec * audio.sample_rate must be >= 1 sample")
    if frame_overlap_sec < 0:
        raise ValueError("vad.window_size_sec must be >= audio.frame_step_sec")

    audio = np.zeros((samples_per_frame * max_history_frames), dtype=np.int16)
    status_console = Console(file=sys.stderr)
    logger.info("Starting microphone stream")
    logger.info(
        "Parameters: sample_rate={}, channels={}, frame_step_sec={}, vad_threshold={}, "
        "check_history={}, speech_ratio_threshold={}, background_ratio_threshold={}, "
        "max_history_frames={}, max_segment_sec={}, asr_model_path={}, vad_onnx_path={}, device_id={}",
        samplerate,
        channels,
        frame_step_sec,
        threshold,
        check_history,
        speech_ratio_threshold,
        background_ratio_threshold,
        max_history_frames,
        max_segment_sec,
        config.asr.model_path,
        config.vad.onnx_model_path,
        input_device_id,
    )

    asr = WhisperGenAIASR(
        WhisperGenAIConfig(
            model_path=config.asr.model_path,
            execution_provider=config.asr.execution_provider,
            max_length=config.asr.max_length,
            num_beams=config.asr.num_beams,
            prompt=config.asr.prompt,
        )
    )

    vad = FrameVAD(
        SileroVADConfig(
            onnx_model_path=config.vad.onnx_model_path,
            sample_rate=samplerate,
            threshold=threshold,
            frame_len_sec=frame_step_sec,
            frame_overlap_sec=frame_overlap_sec,
        )
    )
    vad.reset()

    device_list = sd.query_devices()
    logger.info("Detected audio devices:\n{}", device_list)

    length = samples_per_frame

    vad_state = "Silence"
    asr_state = "Idle"
    with Live(build_status_text(vad_state, asr_state), console=status_console, refresh_per_second=20) as live:

        def log_with_break(level: str, message: str, *args):
            live.console.file.write("\n")
            live.console.file.flush()
            logger.log(level, message, *args)

        async for indata, status in inputstream_generator(
            channels=channels,
            samplerate=samplerate,
            chunk_size=samples_per_frame,
            device_id=input_device_id,
        ):
            if status:
                log_with_break("ERROR", "sounddevice status: {}", status)

            do_decode = False

            signal = np.frombuffer(indata, dtype=np.int16)
            signal_float = signal.astype(np.float32) / 32768.0
            try:
                text = vad.transcribe(signal_float)
            except Exception as exc:
                log_with_break("ERROR", "VAD error: {}", exc)
                continue

            if not text:
                log_with_break("WARNING", "Empty VAD output. Skip this frame.")
                continue

            write_index = min(speech_count, max_history_frames - 1)
            audio[write_index * length : (write_index + 1) * length] = signal

            speech_histories.append(text[0])
            if len(speech_histories) > check_history:
                del speech_histories[0]
            history_len = len(speech_histories)
            speech_count_in_history = speech_histories.count(1)
            background_count_in_history = speech_histories.count(0)
            required_speech_count = math.ceil(history_len * speech_ratio_threshold)
            required_background_count = math.ceil(history_len * background_ratio_threshold)
            vad_state = "Speech" if text[0] == 1 else "Silence"
            live.update(build_status_text(vad_state, asr_state), refresh=True)

            if is_speech and speech_count >= max_history_frames:
                log_with_break("WARNING", "Speaking and {:.2f} seconds have passed", max_segment_sec)
                do_decode = True
            elif text[0] == 1:
                speech_count += 1
                if (
                    not is_speech
                    and history_len >= min_history_for_decision
                    and speech_count_in_history >= required_speech_count
                ):
                    is_speech = True
            else:
                if history_len >= min_history_for_decision and background_count_in_history >= required_background_count:
                    if is_speech:
                        do_decode = True
                        speech_count += 1
                    else:
                        audio[:] = 0
                        speech_count = 0
                    is_speech = False

            if do_decode:
                asr_state = "Decoding"
                live.update(build_status_text(vad_state, asr_state), refresh=True)
                valid_samples = speech_count * length
                if valid_samples > 0:
                    audio_segment = audio[:valid_samples].astype(np.float32) / 32768.0
                    try:
                        result = asr.transcribe(audio_segment, samplerate)
                        log_with_break("INFO", "Recognized text: {}", result)
                    except Exception as exc:
                        log_with_break("ERROR", "ASR decode error: {}", exc)

                audio[:] = 0
                speech_count = 0
                asr_state = "Idle"
                live.update(build_status_text(vad_state, asr_state), refresh=True)


async def main(config_path: Path):
    config = load_config(config_path)
    configure_logger(config.logging.level)
    logger.info("Application started with config={}", config_path)
    audio_task = asyncio.create_task(mic_stream(config))
    await audio_task


if __name__ == "__main__":
    try:
        args = parse_args()
        asyncio.run(main(args.config))
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        sys.exit(0)
