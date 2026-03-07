# online_asr_with_vad

Online speech recognition implementation using PyTorch, NeMo, and `openai-whisper`.

## What This Example Does

`online_asr_with_vad.py` runs the following pipeline:

1. Capture microphone audio in short chunks.
2. Run frame-level VAD using NeMo `vad_marblenet`.
3. Buffer audio while speech is active.
4. Decode with Whisper when speech ends or when max segment length is reached.
5. Print recognition results to logs.

## Structure

- `online_asr_with_vad.py`: Main loop
- `vad.py`: NeMo VAD wrapper
- `data_loader.py`: Data layer used for NeMo inference
- `config.yaml`: Runtime configuration
- `../doc/configuration.md`: Configuration reference
- `../utils/sound_device_list.py`: Audio device listing utility
- `pyproject.toml`: Dependency definition
- `uv.lock`: Lock file

## Setup (uv)

```bash
cd online_asr_with_vad
uv venv
uv sync
```

## Run

```bash
uv run python online_asr_with_vad.py
```

With an explicit config file:

```bash
uv run python online_asr_with_vad.py --config config.yaml
```

## Audio Device Selection

```bash
python ../utils/sound_device_list.py
```

Set the selected device ID in `config.yaml` under `audio.input_device_id`.
