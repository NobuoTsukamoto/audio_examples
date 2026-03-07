# tools

## export_silero_vad_onnx.py

Tool to export Silero VAD (PyTorch) to ONNX.  
The exported ONNX signature is standardized as:

- Inputs: `input(float32[B,T])`, `sr(int64[1])`, `state(float32[2,B,128])`
- Outputs: `output(...)`, `state_out(float32[2,B,128])`

### Prerequisites

This tool requires `torch` only at export time.

```powershell
uv pip install --python online_asr_with_vad_onnx\.venv\Scripts\python.exe torch onnxruntime
```

### Example

```powershell
uv run --python online_asr_with_vad_onnx\.venv\Scripts\python.exe python tools/export_silero_vad_onnx.py
```

The output file is written to `online_asr_with_vad_onnx/models/silero_vad.onnx` by default.
