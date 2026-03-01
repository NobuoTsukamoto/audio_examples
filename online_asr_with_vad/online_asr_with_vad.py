#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2023 Nobuo Tsukamoto
This software is released under the MIT License.
See the LICENSE file in the project root for more information.
"""

import asyncio
import math
import os
import sys

import numpy as np
import sounddevice as sd
from loguru import logger

from vad import FrameVAD
import whisper


def configure_logger():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=log_level)


async def inputstream_generator(channels, samplerate, chunk_size):
    """Generator that yields blocks of input data as NumPy arrays."""
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
        device=0,
    )
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


async def mic_stream():
    """Run online VAD and decode with Whisper at utterance boundaries."""
    window_size = 0.31
    step = 0.01
    threshold = 0.5
    samplerate = 16000
    channels = 1

    speech_count = 0
    is_speech = False
    speech_histories = []
    max_history = 3000  # 30 seconds
    check_history = 200  # 2 seconds
    min_history_for_decision = 20  # 200 ms
    speech_ratio_threshold = 0.8
    background_ratio_threshold = 0.4

    audio = np.zeros((160 * max_history), dtype=np.int16)
    logger.info("Starting microphone stream")
    logger.info(
        "Parameters: sample_rate={}, channels={}, step={}, vad_threshold={}, "
        "check_history={}, speech_ratio_threshold={}, background_ratio_threshold={}, "
        "max_history={}, whisper_model={}",
        samplerate,
        channels,
        step,
        threshold,
        check_history,
        speech_ratio_threshold,
        background_ratio_threshold,
        max_history,
        "base",
    )

    model = whisper.load_model("base")
    options = whisper.DecodingOptions()

    vad = FrameVAD(
        sample_rate=samplerate,
        threshold=threshold,
        frame_len=step,
        frame_overlap=(window_size - step) / 2,
    )
    vad.reset()

    device_list = sd.query_devices()
    logger.info("Detected audio devices:\n{}", device_list)

    length = int(step * samplerate)

    async for indata, status in inputstream_generator(
        channels=channels,
        samplerate=samplerate,
        chunk_size=int(step * samplerate),
    ):
        if status:
            logger.error("sounddevice status: {}", status)

        do_decode = False

        signal = np.frombuffer(indata, dtype=np.int16)
        try:
            text = vad.transcribe(signal)
        except Exception as exc:
            logger.error("VAD error: {}", exc)
            continue

        if not text:
            logger.warning("Empty VAD output. Skip this frame.")
            continue

        # copy to buffer
        write_index = min(speech_count, max_history - 1)
        audio[write_index * length : (write_index + 1) * length] = signal

        logger.debug(
            "VAD result={}, speech_count={}, history_len={}",
            text[0],
            speech_count,
            len(speech_histories),
        )
        speech_histories.append(text[0])
        if len(speech_histories) > check_history:
            del speech_histories[0]
        history_len = len(speech_histories)
        speech_count_in_history = speech_histories.count(1)
        background_count_in_history = speech_histories.count(0)
        required_speech_count = math.ceil(history_len * speech_ratio_threshold)
        required_background_count = math.ceil(history_len * background_ratio_threshold)

        # Speaking and 30 seconds have passed
        if is_speech and speech_count >= max_history:
            logger.warning("Speaking and 30 seconds have passed")
            do_decode = True

        # speech
        elif text[0] == 1:
            speech_count += 1

            if (
                not is_speech
                and history_len >= min_history_for_decision
                and speech_count_in_history >= required_speech_count
            ):
                is_speech = True
                logger.debug(
                    "Detect speech {} / {}",
                    speech_count_in_history,
                    history_len,
                )

        # background
        else:
            if (
                history_len >= min_history_for_decision
                and background_count_in_history >= required_background_count
            ):
                if is_speech:
                    logger.debug(
                        "Detect background and transcribe. {} / {}",
                        background_count_in_history,
                        history_len,
                    )

                    do_decode = True
                    speech_count += 1
                else:
                    audio[:] = 0
                    speech_count = 0

                is_speech = False

        if do_decode:
            valid_samples = speech_count * length
            if valid_samples > 0:
                audio_segment = audio[:valid_samples].astype(np.float32) / 32768.0
                audio_segment = whisper.pad_or_trim(audio_segment)
                mel = whisper.log_mel_spectrogram(audio_segment).to(model.device)
                try:
                    result = whisper.decode(model, mel, options)
                    logger.info("Recognized text: {}", result.text)
                except Exception as exc:
                    logger.error("Whisper decode error: {}", exc)

            audio[:] = 0
            speech_count = 0


async def main():
    configure_logger()
    logger.info("Application started")
    audio_task = asyncio.create_task(mic_stream())
    await audio_task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        sys.exit(0)
