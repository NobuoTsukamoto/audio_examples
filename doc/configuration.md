# Configuration Specification

`online_asr_with_vad` reads runtime parameters from a YAML file.

- Default path: `online_asr_with_vad/config.yaml`
- Override path: `--config <path>`

Example:

```bash
python online_asr_with_vad/online_asr_with_vad.py --config online_asr_with_vad/config.yaml
```

## YAML Schema

```yaml
audio:
  input_device_id: <int>
  sample_rate: <int>
  channels: <int>
  frame_step_sec: <float>

vad:
  model_name: <string>
  window_size_sec: <float>
  threshold: <float>
  check_history_frames: <int>
  min_history_for_decision_frames: <int>
  speech_ratio_threshold: <float>
  background_ratio_threshold: <float>

segment:
  max_history_frames: <int>

whisper:
  model_name: <string>

logging:
  level: <string>
```

## Parameter Details

### `audio`

- `input_device_id`: Input device id used by `sounddevice.InputStream`.
- `sample_rate`: Microphone sample rate in Hz.
- `channels`: Number of input channels (typically `1`).
- `frame_step_sec`: Frame interval in seconds for streaming and VAD updates.
- `frame_step_sec` must satisfy `int(sample_rate * frame_step_sec) >= 1`.

### `vad`

- `model_name`: NeMo VAD pretrained model name (for example `vad_marblenet`).
- `window_size_sec`: Analysis window size in seconds.
- `threshold`: Speech decision threshold (`probs[1] >= threshold`).
- `threshold` valid range is `0.0..1.0`.
- `check_history_frames`: Rolling history size for state decisions.
- `min_history_for_decision_frames`: Minimum history required before ratio-based transitions.
- `speech_ratio_threshold`: Ratio required to enter speech state.
- `background_ratio_threshold`: Ratio required to exit speech state.
- `speech_ratio_threshold` and `background_ratio_threshold` valid range is `0.0..1.0`.
- Recommended: `check_history_frames >= min_history_for_decision_frames`.

### `segment`

- `max_history_frames`: Max buffered frames before forced decode.

### `whisper`

- `model_name`: Whisper model name (for example `base`).

### `logging`

- `level`: Log level for `loguru` (`DEBUG`, `INFO`, `WARNING`, `ERROR`, ...).
- `LOG_LEVEL` environment variable takes precedence over this value.

## Derived Values

These values are not stored in YAML and are computed at runtime:

- `samples_per_frame = int(audio.sample_rate * audio.frame_step_sec)`
- `max_segment_sec = segment.max_history_frames * audio.frame_step_sec`
- `frame_overlap_sec = (vad.window_size_sec - audio.frame_step_sec) / 2`

## Validation Rules

- `int(audio.sample_rate * audio.frame_step_sec) >= 1`
- `vad.window_size_sec >= audio.frame_step_sec`

If either rule fails, the app raises `ValueError` at startup.

## Default Values

Current default values in `online_asr_with_vad/config.yaml`:

- `audio.input_device_id: 0`
- `audio.sample_rate: 16000`
- `audio.channels: 1`
- `audio.frame_step_sec: 0.01`
- `vad.model_name: vad_marblenet`
- `vad.window_size_sec: 0.31`
- `vad.threshold: 0.5`
- `vad.check_history_frames: 200`
- `vad.min_history_for_decision_frames: 20`
- `vad.speech_ratio_threshold: 0.8`
- `vad.background_ratio_threshold: 0.4`
- `segment.max_history_frames: 3000`
- `whisper.model_name: base`
- `logging.level: INFO`
