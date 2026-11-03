# whisper_engine_apple.py
# Opt-in macOS 26+ backend (whisper.backend: apple) using Apple's on-device
# SpeechTranscriber, the model behind system Dictation. Talks to the Swift
# helper platform/macos/apple_stt.swift, compiled on first use with the Xcode
# Command Line Tools. Mirrors WhisperEngine's public API.

import json
import logging
import os
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .utils import get_user_app_data_path

HELPER_SOURCE = Path(__file__).parent / 'platform' / 'macos' / 'apple_stt.swift'


# ---- Helper build -----------------------------------------------------------

# Compiles the Swift helper into ~/.whisperkey/bin, rebuilding when the source
# is newer (e.g. after upgrading whisper-local).
def ensure_helper() -> Path:
    binary = Path(get_user_app_data_path()) / 'bin' / 'apple-stt'
    if binary.exists() and binary.stat().st_mtime >= HELPER_SOURCE.stat().st_mtime:
        return binary

    swiftc = shutil.which('swiftc')
    if not swiftc:
        raise RuntimeError("apple backend needs the Swift compiler. "
                           "Install it with: xcode-select --install")
    binary.parent.mkdir(parents=True, exist_ok=True)
    tmp = binary.with_suffix('.tmp')
    print("🔨 Building Apple speech helper (first run only)...")
    result = subprocess.run([swiftc, '-O', str(HELPER_SOURCE), '-o', str(tmp)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Building apple-stt failed (needs macOS 26+):\n{result.stderr.strip()}")
    os.replace(tmp, binary)
    return binary


# ---- Engine -----------------------------------------------------------------

class WhisperEngineApple:
    def __init__(self,
                 model_key: str = "apple",
                 device: str = "cpu",
                 compute_type: str = "int8",
                 language: Optional[str] = None,
                 beam_size: int = 5,
                 initial_prompt: str = "",
                 hotwords: Optional[list] = None,
                 task: str = "transcribe",
                 vad_manager=None,
                 model_registry=None,
                 log_transcriptions: bool = False,
                 languages: Optional[list] = None):

        # Whisper-only knobs (model, device, beam, prompt, hotwords) are accepted
        # so callers stay backend-agnostic, but Apple's model has no equivalent.
        # ponytail: hotwords ignored; SpeechTranscriber contextual strings if vocab bias matters
        self.model_key = model_key
        self.device = device
        self.compute_type = compute_type
        self.language = None if language in (None, 'auto') else language
        self.initial_prompt = initial_prompt or None
        self.hotwords = hotwords or []
        self.task = task
        self.vad_manager = vad_manager
        self.log_transcriptions = log_transcriptions
        self.allowed_languages = []
        self.last_language_outsider = None
        self.logger = logging.getLogger(__name__)
        if task == 'translate':
            self.logger.warning("apple backend cannot translate; transcribing instead")

        self._lock = threading.Lock()
        self._process = None
        self._binary = ensure_helper()
        self._start_helper()

    # Apple picks one locale per helper; auto-detect isn't offered, so default to US English.
    def _locale(self) -> str:
        return self.language or 'en-US'

    def _start_helper(self):
        print(f"🧠 Starting Apple on-device speech model [{self._locale()}]...")
        self._process = subprocess.Popen([str(self._binary), self._locale()],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        # First run per locale downloads the model from Apple before "ready".
        reply = self._read_reply()
        if 'error' in reply:
            raise RuntimeError(f"Apple speech helper: {reply['error']}")
        print(f"   ✓ Apple speech model [{reply.get('locale')}] ready!")

    def _read_reply(self) -> dict:
        # ponytail: blocking readline with no timeout; add one if the helper ever wedges
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("Apple speech helper exited")
        return json.loads(line)

    def _round_trip(self, audio: np.ndarray) -> dict:
        payload = audio.astype('<f4').tobytes()
        self._process.stdin.write(struct.pack('<I', len(audio)) + payload)
        self._process.stdin.flush()
        return self._read_reply()

    # ---- Public API (mirrors WhisperEngine) ---------------------------------

    def transcribe_audio(self, audio_data: Optional[np.ndarray]) -> Optional[str]:
        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("No audio data to transcribe")
            return None

        try:
            if self.vad_manager and self.vad_manager.is_available():
                if not self.vad_manager.check_audio_for_speech(audio_data):
                    print("   ✗ No speech detected, skipping transcription")
                    return None

            audio = audio_data.flatten().astype(np.float32)
            t0 = time.time()
            with self._lock:
                try:
                    reply = self._round_trip(audio)
                except (BrokenPipeError, RuntimeError):
                    # Helper crashed between dictations: restart once and retry.
                    self.logger.warning("Apple speech helper died; restarting")
                    self._start_helper()
                    reply = self._round_trip(audio)

            if 'error' in reply:
                raise RuntimeError(reply['error'])
            text = reply.get('text', '')
            print(f"   ✓ Transcription completed in {time.time() - t0:.1f} seconds")

            if self.log_transcriptions:
                self.logger.info(f"Transcribed: '{text}'")
            else:
                self.logger.info(f"Transcribed {len(text)} chars")
            return text or None

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}", exc_info=True)
            print(f"❌ Transcription failed: {e}")
            return None

    # Apple ships a single model per locale, so there is nothing to switch to.
    def change_model(self,
                     new_model_key: str,
                     progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        return False

    def is_loading(self) -> bool:
        return False
