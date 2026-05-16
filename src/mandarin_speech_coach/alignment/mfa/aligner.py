import os
import shutil
import subprocess
import tempfile

import gradio as gr
import librosa
import soundfile as sf

from praatio import textgrid

from mandarin_speech_coach.alignment.base import BaseAligner
from mandarin_speech_coach.core.types import (
    AlignmentResult,
    Segment,
)

MFA_ACOUSTIC_MODEL = os.environ.get(
    "MFA_ACOUSTIC_MODEL",
    "mandarin_mfa"
)

MFA_DICTIONARY = os.environ.get(
    "MFA_DICTIONARY",
    "mandarin_mfa"
)


class MFAAligner(BaseAligner):
    def align(self, audio_path: str, text: str):
        if not shutil.which("mfa"):
            raise gr.Error(
                "Montreal Forced Aligner (`mfa`) is not installed or not on PATH. "
                "Activate your MFA conda env in the same shell as the app. See README.md."
            )

        # Whitespace-separated tokens.
        # Mandarin MFA models expect segmented orthography.
        mfa_text = " ".join(ch for ch in text if not ch.isspace())

        # Prepare audio for MFA.
        # 16 kHz mono 16-bit PCM WAV (Kaldi/MFA requirement).
        y, _ = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        wav_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ).name

        sf.write(
            wav_path,
            y,
            16000,
            subtype="PCM_16"
        )

        corpus_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()

        try:
            speaker_dir = os.path.join(corpus_dir, "speaker")
            os.makedirs(speaker_dir)

            shutil.copy(
                wav_path,
                os.path.join(speaker_dir, "utt1.wav")
            )

            with open(
                os.path.join(speaker_dir, "utt1.lab"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(mfa_text)

            # --no_tokenization: we already pass space-separated characters (see _format_mfa_transcript)
            # Batch align first (more reliable for mandarin_mfa than align_one; see MFA issue #908)
            cmd = [
                "mfa",
                "align",
                "--clean",
                "--single_speaker",
                "--no_tokenization",
                "-j",
                "1",
                corpus_dir,
                MFA_DICTIONARY,
                MFA_ACOUSTIC_MODEL,
                output_dir,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise gr.Error(result.stderr)

            tg_path = os.path.join(
                output_dir,
                "speaker",
                "utt1.TextGrid",
            )
            if not os.path.exists(tg_path):
                tg_path = os.path.join(
                    output_dir,
                    "utt1.TextGrid"
                )

            # Read MFA word intervals from a Praat TextGrid (praatio 6.x API).
            tg = textgrid.openTextgrid(
                tg_path,
                includeEmptyIntervals=True
            )

            tier_name = "words" if "words" in tg.tierNames else None
            if tier_name is None:
                for name in tg.tierNames:
                    if "word" in name.lower():
                        tier_name = name
                        break

            segments = []
            if tier_name:
                tier = tg.getTier(tier_name)

                for entry in tier.entries:
                    if entry.label.strip():
                        segments.append(
                            Segment(
                                label=entry.label,
                                start=entry.start,
                                end=entry.end,
                            )
                        )

            return AlignmentResult(
                method="mfa",
                segments=segments,
            )

        finally:
            shutil.rmtree(corpus_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

            if os.path.exists(wav_path):
                os.remove(wav_path)
