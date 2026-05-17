import torch
import torchaudio
import numpy as np

from mandarin_speech_coach.alignment.base import BaseAligner
from mandarin_speech_coach.core.types import (
    AlignmentResult,
    Segment,
)

from mandarin_speech_coach.models.registry import (
    get_ctc,
    get_device,
)


class CTCAligner(BaseAligner):
    def __init__(self):
        self.processor, self.model = get_ctc()
        self.device = get_device()

    def align(self, audio_path: str, text: str):
        # 1. Load Audio and text
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=16000
            )(waveform)

        # 2. Get Emissions
        input_values = self.processor(
            waveform.squeeze(),
            return_tensors="pt",
            sampling_rate=16000
        ).input_values.to(self.device)

        labels = self.processor(
            text=text,
            return_tensors="pt"
        ).input_ids[0]

        with torch.no_grad():
            logits = self.model(input_values).logits

        emissions = torch.log_softmax(
            logits,
            dim=-1
        )[0].cpu()

        # 3. Log-prior smoothing to improve alignment robustness
        log_priors = torch.log(
            torch.bincount(labels, minlength=emissions.size(1)).float() + 1e-8
        )
        alpha = 0.5
        emissions = emissions - alpha * log_priors.unsqueeze(0)

        # 4. Generate Trellis (dynamic programming table)
        blank_id = self.processor.tokenizer.pad_token_id

        # Insert blank tokens between characters for CTC
        token_path = [blank_id] + [
            val
            for t in labels
            for val in (t, blank_id)
        ]

        trellis = torch.full(
            (emissions.shape[0], len(token_path)),
            -float("inf")
        )
        trellis[0, 0] = emissions[0, blank_id]
        trellis[0, 1] = emissions[0, token_path[1]]

        for t in range(1, emissions.shape[0]):
            for j in range(len(token_path)):
                # Case 1: Stay at the same token (can be blank or a character)
                prev_trellis = trellis[t - 1, j]

                # Case 2: Move from the previous token
                if j > 0:
                    prev_trellis = max(
                        prev_trellis,
                        trellis[t - 1, j - 1]
                    )

                # Case 3: Skip a blank token
                if (
                    j > 1
                    and token_path[j] != blank_id
                    and token_path[j - 2] == token_path[j]
                ):
                    prev_trellis = max(
                        prev_trellis,
                        trellis[t - 1, j - 2]
                    )

                trellis[t, j] = (
                    prev_trellis
                    + emissions[t, token_path[j]]
                )

        # 5. Backtrack to find the most likely path
        path = []
        j = trellis.shape[1] - 1

        for t in range(trellis.shape[0] - 1, -1, -1):
            # Find the index of the maximum predecessor in the trellis
            if (
                j > 1
                and token_path[j] != blank_id
                and token_path[j - 2] == token_path[j]
                and trellis[t - 1, j - 2] >= trellis[t - 1, j - 1]
                and trellis[t - 1, j - 2] >= trellis[t - 1, j]
            ):
                j = j - 2

            elif (
                j > 0
                and trellis[t - 1, j - 1] >= trellis[t - 1, j]
            ):
                j = j - 1

            path.append((token_path[j], t, torch.exp(emissions[t, token_path[j]]).item()))

        path.reverse()

        # 6. Merge and Format Segments
        segments = []
        prev_token = None

        for token, time_idx, score in path:
            if token != blank_id:
                char = self.processor.decode(token)

                # Start a new segment if:
                # - this is the first token
                # - character changed
                # - previous path token was blank
                if (
                    not segments
                    or segments[-1]["char"] != char
                    or prev_token == blank_id
                ):
                    segments.append(
                        {
                            "char": char,
                            "start_frame": time_idx,
                            "end_frame": time_idx,
                            "scores": [score],
                        }
                    )
                else:
                    segments[-1]["end_frame"] = time_idx
                    segments[-1]["scores"].append(score)

            prev_token = token

        # Convert frame indices to seconds
        ratio = waveform.shape[1] / emissions.shape[0] / 16000
        output_segments = [
            Segment(
                label=seg["char"],
                start=round(seg["start_frame"] * ratio, 3),
                end=round((seg["end_frame"] + 1) * ratio, 3),
                confidence=round(np.mean(seg['scores']), 3),
            )
            for seg in segments
        ]

        return AlignmentResult(
            method="ctc",
            segments=output_segments,
        )
