#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import asyncio
import sys
from datetime import datetime

import numpy as np
import sounddevice as sd

from vad import FrameVAD
import whisper


async def inputstream_generator(channels, samplerate, chunck_size):
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(
        callback=callback,
        channels=channels,
        samplerate=samplerate,
        blocksize=chunck_size,
        dtype="int16",
        device=0,
    )
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


async def mic_stream():
    """Show minimum and maximum value of each incoming audio block."""
    window_size = 0.31
    step = 0.01
    threshold = 0.5
    samplerate = 16000
    channels = 1
    window = 200
    downsample = 10

    speech_count = 0
    is_speech = False
    speech_histories = []
    max_history = 3000  # 30 seconds
    check_history = 200  # 2 seconds
    speech_threshold = int(check_history * 0.8)
    background_threshold = int(check_history * 0.4)

    audio = np.zeros((160 * max_history), dtype=np.int16)
    model = whisper.load_model("base")
    options = whisper.DecodingOptions()

    vad = FrameVAD(
        sample_rate=samplerate,
        threshold=threshold,
        frame_len=step,
        frame_overlap=(window_size - step) / 2,
        offset=0,
    )
    vad.reset()

    device_list = sd.query_devices()
    print(device_list)

    # length = int(window * samplerate / (1000 * downsample))
    lenght = int(step * samplerate)

    async for indata, status in inputstream_generator(
        channels=channels,
        samplerate=samplerate,
        chunck_size=int(step * samplerate),
    ):
        if status:
            print(status)

        do_decode = False

        signal = np.frombuffer(indata, dtype=np.int16)
        text = vad.transcribe(signal)

        # copy to buffer
        audio[speech_count * lenght : (speech_count + 1) * lenght] = signal

        print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], text[0])
        speech_histories.append(text[0])
        if len(speech_histories) > check_history:
            del speech_histories[0]
        else:
            continue

        # Speaking and 30 seconds have passed
        if is_speech and speech_count >= max_history:
            print("Speaking and 30 seconds have passed.")
            do_decode = True

        # speech
        elif text[0] == 1:
            speech_count += 1

            if not is_speech and speech_histories.count(1) > speech_threshold:
                is_speech = True
                print(
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "detect speech {} / {}".format(
                        speech_histories.count(1), len(speech_histories)
                    ),
                )

        # background
        else:
            if speech_histories.count(0) > background_threshold:
                if is_speech:
                    print(
                        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        "detect background and transcribe. {} / {}".format(
                            speech_histories.count(0), len(speech_histories)
                        ),
                    )

                    do_decode = True
                    speech_count += 1
                else:
                    audio[:] = 0
                    speech_count = 0

                is_speech = False

        if do_decode:
            audio = audio.flatten().astype(np.float32) / 32768.0
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            result = whisper.decode(model, mel, options)
            print(result)

            audio[:] = 0
            speech_count = 0


async def main():
    audio_task = asyncio.create_task(mic_stream())
    await audio_task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
