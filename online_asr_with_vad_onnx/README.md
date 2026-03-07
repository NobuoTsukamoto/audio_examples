# online_asr_with_vad_onnx

Online speech recognition implementation using `onnxruntime-genai` Whisper ASR and `onnxruntime` Silero ONNX VAD.  
The original `online_asr_with_vad` implementation is kept unchanged.

## 1. Create a Virtual Environment (uv)

```powershell
cd online_asr_with_vad_onnx
uv venv .venv
```

Install dependencies:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## 2. Prepare Models

Place the following files based on default paths in `config.yaml`:

- `./models/silero_vad.onnx`
- `./models/whisper/` (onnxruntime-genai Whisper model assets)

If Silero ONNX is not prepared yet, generate it with the root-level tool:

```powershell
uv run --python .venv\Scripts\python.exe python ..\tools\export_silero_vad_onnx.py
```

## 3. Configuration

Adjust the following fields in `config.yaml` for your environment:

- `audio.input_device_id`
- `vad.onnx_model_path`
- `asr.model_path`
- optionally `asr.execution_provider` (`cpu` or `cuda`)

## 4. Run

```powershell
uv run --python .venv\Scripts\python.exe python online_asr_with_vad_onnx.py --config config.yaml
```
