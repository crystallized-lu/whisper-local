# system_audio.py
# Optional system-audio (loopback) capture — transcribe what you HEAR (meetings,
# videos, calls), not just the microphone. Deliberately isolated from
# AudioRecorder's always-on mic stream so it cannot affect normal dictation.
#
# Backed by the optional `soundcard` package (WASAPI loopback on Windows,
# CoreAudio on macOS), installed via `pip install whisper-local[loopback]`.
# soundcard is imported lazily so the base install never depends on it.
#
# EXPERIMENTAL: the actual capture path depends on OS loopback support and needs
# real-hardware verification — it can't be exercised in CI (no audio device).

import logging

import numpy as np

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000

_INSTALL_HINT = (
    "System-audio capture needs the optional 'loopback' extra:\n"
    "    pip install whisper-local[loopback]"
)


# True when the soundcard backend is importable. Cheap enough to call before
# offering the feature in a menu / CLI.
def is_available() -> bool:
    try:
        import soundcard  # noqa: F401
        return True
    except Exception:
        return False


# Record `seconds` of the default speaker's output as a mono float32 array at
# 16 kHz (the rate the Whisper pipeline expects). Returns None — never raises —
# if soundcard is missing or no loopback endpoint is available, so callers can
# degrade gracefully.
def capture_loopback(seconds: float, samplerate: int = WHISPER_SAMPLE_RATE):
    try:
        import soundcard as sc
    except Exception:
        logger.warning("soundcard not installed; cannot capture system audio")
        print(_INSTALL_HINT)
        return None

    try:
        speaker = sc.default_speaker()
        # A loopback "microphone" mirrors what the speaker is playing.
        loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        frames = int(samplerate * max(0.1, float(seconds)))
        with loopback.recorder(samplerate=samplerate, channels=1) as rec:
            data = rec.record(numframes=frames)
    except Exception as e:
        logger.error(f"System-audio capture failed: {e}")
        print(f"System-audio capture failed: {e}")
        return None

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:  # collapse any stray channel dimension to mono
        arr = arr.mean(axis=1)
    return arr


# Capture then transcribe in one step, reusing the app's existing Whisper engine
# and its transcribe_audio contract (mono float32 @ 16 kHz). Returns '' on any
# failure or silence.
def record_and_transcribe(seconds: float, whisper_engine,
                          samplerate: int = WHISPER_SAMPLE_RATE) -> str:
    audio = capture_loopback(seconds, samplerate)
    if audio is None or len(audio) == 0:
        return ''
    return whisper_engine.transcribe_audio(audio) or ''


# CLI entry point for `whisper-local --transcribe-system [SECONDS]`. Records that
# many seconds of system audio, transcribes with the configured model, prints the
# text and copies it to the clipboard. Self-contained: bootstraps its own engine
# (mirrors selftest) so it doesn't require the full app to be running. Returns a
# process exit code.
def run_cli(seconds: int = 10) -> int:
    if not is_available():
        print(_INSTALL_HINT)
        return 1

    try:
        from .config_manager import ConfigManager
        from .model_registry import ModelRegistry
        from .voice_activity_detection import VadManager
        from .whisper_engine import WhisperEngine

        cm = ConfigManager(quiet=True)
        whisper_cfg = cm.get_whisper_config()
        registry = ModelRegistry(
            whisper_models_config=whisper_cfg.get('models', {}),
            streaming_models_config={},
        )
        vad_manager = VadManager(
            vad_precheck_enabled=False, vad_realtime_enabled=False,
            vad_onset_threshold=0.5, vad_offset_threshold=0.5,
            vad_min_speech_duration=0.1, vad_silence_timeout_seconds=999.0,
        )
        # Honour the configured backend so we don't silently transcribe with
        # faster-whisper (and trigger a redundant model download) when the user
        # chose whisper_cpp — mirrors selftest's end-to-end bootstrap.
        engine_kwargs = dict(
            model_key=whisper_cfg['model'], device=whisper_cfg['device'],
            compute_type=whisper_cfg['compute_type'],
            language=whisper_cfg.get('language', 'auto'),
            beam_size=whisper_cfg.get('beam_size', 5), initial_prompt='',
            hotwords=[], task='transcribe', vad_manager=vad_manager,
            model_registry=registry,
        )
        backend = whisper_cfg.get('backend', 'faster_whisper')
        if backend == 'apple':
            from .whisper_engine_apple import WhisperEngineApple
            engine = WhisperEngineApple(**engine_kwargs)
        elif backend == 'whisper_cpp':
            from .whisper_engine_cpp import WhisperEngineCpp
            engine = WhisperEngineCpp(**engine_kwargs)
        else:
            engine = WhisperEngine(**engine_kwargs)
    except Exception as e:
        print(f"Could not initialise transcription engine: {e}")
        return 1

    print(f"🔊 Capturing {seconds}s of system audio — play the audio now...")
    text = record_and_transcribe(seconds, engine)
    if not text:
        print("No speech detected in the captured system audio.")
        return 1

    print("\n--- Transcription ---")
    print(text)
    try:
        import pyperclip
        pyperclip.copy(text)
        print("\n(Copied to clipboard.)")
    except Exception:
        pass
    return 0
