import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_WAV = REPO_ROOT / "samples" / "example.wav"
TARGET_TEXT = "我喜欢机器学习"


@pytest.fixture(scope="module")
def example_wav_path():
    if not EXAMPLE_WAV.is_file():
        pytest.fail(f"Missing sample audio: {EXAMPLE_WAV}")
    return str(EXAMPLE_WAV)


@pytest.fixture(scope="module")
def target_text():
    return TARGET_TEXT


@pytest.fixture(scope="module")
def require_mfa():
    if not shutil.which("mfa"):
        pytest.skip(
            "Montreal Forced Aligner (`mfa`) is not on PATH. "
            "Run `conda activate aligner` in the same shell as pytest."
        )
