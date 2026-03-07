# Audio Examples

This repository contains local speech recognition examples.
Two implementations are currently maintained side-by-side.

## Implementations

1. `online_asr_with_vad`  
Original implementation based on PyTorch, NeMo, and `openai-whisper`.  
Details: [`online_asr_with_vad/README.md`](online_asr_with_vad/README.md)

2. `online_asr_with_vad_onnx`  
Implementation based on `onnxruntime-genai` (Whisper) and `onnxruntime` (Silero ONNX VAD).  
Details: [`online_asr_with_vad_onnx/README.md`](online_asr_with_vad_onnx/README.md)

## Repository Layout

- `online_asr_with_vad/`: PyTorch-based implementation
- `online_asr_with_vad_onnx/`: ONNX-based implementation
- `tools/`: Conversion and helper tools (for example, Silero ONNX export)
- `doc/`: Shared documentation
- `utils/`: Shared utilities such as audio device listing

## Notes

- Python environment files are managed within each implementation directory.
- Environment-related files for `online_asr_with_vad` were moved to `online_asr_with_vad/`.
