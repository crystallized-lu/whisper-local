# test_whisper_engine_apple.py
# Exercises WhisperEngineApple's stdin/stdout framing and crash recovery against
# a fake helper that speaks the apple_stt.swift protocol, so it runs anywhere
# (no macOS 26, no Speech framework, no Swift compiler needed).

import sys
import tempfile
import textwrap
import unittest
import unittest.mock as mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from whisper_key import whisper_engine_apple  # noqa: E402

# Echoes the sample count and first sample; exits once if the crash marker exists.
FAKE_HELPER = textwrap.dedent('''
    import array, json, os, struct, sys
    marker = os.environ["CRASH_MARKER"]
    print(json.dumps({"ready": True, "locale": sys.argv[1]}), flush=True)
    while True:
        header = sys.stdin.buffer.read(4)
        if len(header) < 4:
            break
        count = struct.unpack("<I", header)[0]
        samples = array.array("f")
        samples.frombytes(sys.stdin.buffer.read(count * 4))
        if os.path.exists(marker):
            os.remove(marker)
            sys.exit(1)
        print(json.dumps({"text": f"{count} samples, first={samples[0]:.2f}"}), flush=True)
''')


class AppleEngineProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        helper = self.tmp / "apple-stt"
        helper.write_text(f"#!{sys.executable}\n{FAKE_HELPER}")
        helper.chmod(0o755)
        self.marker = self.tmp / "crash"
        patches = [
            mock.patch.object(whisper_engine_apple, "ensure_helper", return_value=helper),
            mock.patch.dict("os.environ", {"CRASH_MARKER": str(self.marker)}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_round_trip_frames_audio(self):
        engine = whisper_engine_apple.WhisperEngineApple(language="en-GB")
        audio = np.array([0.5, -0.25, 0.0], dtype=np.float64)  # non-float32 on purpose
        self.assertEqual(engine.transcribe_audio(audio), "3 samples, first=0.50")
        self.assertEqual(engine.transcribe_audio(audio[:1]), "1 samples, first=0.50")

    def test_restarts_helper_after_crash(self):
        engine = whisper_engine_apple.WhisperEngineApple()
        self.marker.touch()
        self.assertEqual(engine.transcribe_audio(np.ones(2, dtype=np.float32)),
                         "2 samples, first=1.00")
        self.assertFalse(self.marker.exists())

    def test_empty_audio_skips_helper(self):
        engine = whisper_engine_apple.WhisperEngineApple()
        self.assertIsNone(engine.transcribe_audio(np.array([], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
