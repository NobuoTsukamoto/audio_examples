# Audio Examples

This repository contains local (on-device) speech recognition examples.
The main implementation is `online_asr_with_vad`.

## What This Example Does

`online_asr_with_vad` runs an online microphone pipeline:

1. Capture mic audio in 10 ms chunks.
2. Run frame-level VAD with NeMo `vad_marblenet`.
3. Keep buffering audio while speech is active.
4. Run Whisper only when speech ends (or after max segment length).
5. Log the recognition result to stdout.

This design reduces device load by avoiding continuous Whisper decoding.

## Repository Structure

- `online_asr_with_vad/online_asr_with_vad.py`: Main loop (mic input, VAD logic, Whisper trigger)
- `online_asr_with_vad/vad.py`: Frame VAD wrapper using NeMo pretrained model
- `online_asr_with_vad/data_loader.py`: Minimal data layer for NeMo inference
- `utils/sound_device_list.py`: Print available audio input/output devices
- `Dockerfile`: Example runtime environment

## Requirements

- Python 3.x
- PyTorch
- NVIDIA NeMo
- OpenAI Whisper (`openai/whisper`)
- `sounddevice`
- `loguru`
- OS packages used by this project: `sox`, `libsndfile1`, `ffmpeg`, `portaudio19-dev`

The provided `Dockerfile` uses `nvcr.io/nvidia/nemo:23.03` and installs the rest.

## Setup

### Docker

```bash
docker build -t audio-examples .
```

### Local environment (example)

```bash
pip install nemo_toolkit
pip install git+https://github.com/openai/whisper.git
pip install sounddevice
pip install loguru
```

Install equivalent OS-level audio packages for your platform as needed.

## Run

```bash
python online_asr_with_vad/online_asr_with_vad.py
```

At first run, pretrained models are downloaded:

- Whisper `base`
- NeMo `vad_marblenet`

After that, inference is local/on-device.

## Current Runtime Parameters

These are hardcoded in `mic_stream()` in `online_asr_with_vad.py`:

- Sample rate: `16000`
- Channels: `1`
- Frame step: `0.01` sec (10 ms)
- VAD threshold: `0.5`
- History window: `check_history = 200` (~2 sec)
- Speech start condition: speech ratio > `80%`
- Speech end condition: background ratio > `40%`
- Max utterance length before forced decode: `30` sec (`max_history = 3000`)
- Whisper model: `base`

## Audio Device Selection

List devices:

```bash
python utils/sound_device_list.py
```

The main script currently uses a fixed device:

- `sounddevice.InputStream(..., device=0)`

If needed, change `device=0` in `online_asr_with_vad.py`.

## Notes and Limitations

- VAD + Whisper execution is local/on-device, but initial model download requires network.
- Parameters are not exposed as CLI options yet; tuning is done by editing source.
- Output is logged to stdout only (no file persistence/API layer yet).

## References

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo/tree/main)
- [Whisper](https://github.com/openai/whisper)
