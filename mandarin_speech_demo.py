import gradio as gr
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2ForCTC, Wav2Vec2Processor
import tempfile
import os
import subprocess
import parselmouth
import numpy as np
from pypinyin import pinyin, Style
from praatio import textgrid
import librosa
import soundfile as sf
import shutil
import plotly.graph_objects as go

# --- Global Settings & Model Loading ---

# Use a font that supports Chinese characters for plotting
# Plotly generally handles this well, but this is a good fallback for matplotlib if needed
# matplotlib.rcParams['font.family'] = ['Heiti TC']

# 1. Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 2. Target Phrase
TARGET_TEXT = "我喜欢机器学习"
TARGET_PINYIN_TONES = pinyin(TARGET_TEXT, style=Style.TONE3)


def format_pinyin(text):
    """Full-syllable pinyin with tone marks (one string per character)."""
    hanzi = "".join(ch for ch in (text or "") if "\u4e00" <= ch <= "\u9fff")
    if not hanzi:
        return "—"
    return " ".join(syl[0] for syl in pinyin(hanzi, style=Style.TONE))


TARGET_PINYIN_TEXT = format_pinyin(TARGET_TEXT)

PHRASE_CARD_CSS = """
.phrase-row { align-items: stretch; }
.phrase-card {
    padding: 1.1rem 1.35rem;
    border-radius: 10px;
    height: 100%;
    box-sizing: border-box;
}
.phrase-card-target {
    background: linear-gradient(180deg, #f4f9ff 0%, #eef5fc 100%);
    border: 1px solid #c5d9f0;
}
.phrase-card-heard {
    background: #fafafa;
    border: 1px solid #e0e0e0;
}
.phrase-card-heard.placeholder { color: #888; }
.phrase-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #5a6a7a;
    margin: 0 0 0.65rem 0;
}
.phrase-hanzi {
    font-size: 2.15rem;
    font-weight: 600;
    line-height: 1.35;
    margin: 0 0 0.4rem 0;
    color: #1a1a1a;
}
.phrase-pinyin {
    font-size: 1.2rem;
    line-height: 1.45;
    margin: 0;
    color: #3d4f5f;
}
"""


def phrase_card_html(label, hanzi, pinyin_text, variant="target"):
    hanzi_display = hanzi if hanzi and hanzi != "—" else "—"
    pinyin_display = pinyin_text if pinyin_text and pinyin_text != "—" else "—"
    extra = " placeholder" if variant == "placeholder" else ""
    card_class = "phrase-card-target" if variant == "target" else f"phrase-card-heard{extra}"
    return (
        f'<div class="phrase-card {card_class}">'
        f'<p class="phrase-label">{label}</p>'
        f'<p class="phrase-hanzi">{hanzi_display}</p>'
        f'<p class="phrase-pinyin">{pinyin_display}</p>'
        "</div>"
    )


TARGET_PHRASE_HTML = phrase_card_html(
    "Say this phrase", TARGET_TEXT, TARGET_PINYIN_TEXT, variant="target"
)
HEARD_PLACEHOLDER_HTML = phrase_card_html(
    "What we heard",
    "—",
    "Record audio, then click Analyze",
    variant="placeholder",
)

TONE_COLORS = {
    '1': 'rgba(204, 229, 255, 0.5)',  # Pale Blue
    '2': 'rgba(204, 255, 204, 0.5)',  # Pale Green
    '3': 'rgba(255, 242, 204, 0.5)',  # Pale Yellow
    '4': 'rgba(255, 204, 204, 0.5)',  # Pale Red
    '5': 'rgba(224, 224, 224, 0.5)',  # Neutral Grey
}

# 3. Load Whisper Model
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(DEVICE)

# 4. Load CTC Alignment Model
ctc_processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
ctc_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn").to(DEVICE)

# 5. MFA configuration (optional; model names from `mfa model download`, overridable via env)
MFA_ACOUSTIC_MODEL = os.environ.get("MFA_ACOUSTIC_MODEL", "mandarin_mfa")
MFA_DICTIONARY = os.environ.get("MFA_DICTIONARY", "mandarin_mfa")


# --- Core Functions ---

def recognize_speech(audio_path):
    """Transcribe audio using Whisper."""
    waveform, sr = librosa.load(audio_path, sr=16000)
    inputs = whisper_processor(waveform, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    
    # Set forced decoder ids for Chinese transcription
    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="chinese", task="transcribe")
    predicted_ids = whisper_model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids)
    
    return whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

def trim_silence(audio_path):
    """Trim leading/trailing silence from an audio file."""
    y, sr = librosa.load(audio_path, sr=16000)
    yt, index = librosa.effects.trim(y, top_db=20)
    trimmed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(trimmed_path, yt, sr)
    return trimmed_path

# --- Alignment Algorithms ---

def ctc_align(audio_path, text):
    """
    Performs forced alignment using a Wav2Vec2-CTC model.
    This is a simplified, self-contained version of the CTC alignment logic.
    """
    # 1. Load Audio and text
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    
    # 2. Get Emissions
    input_values = ctc_processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values.to(DEVICE)
    with torch.no_grad():
        logits = ctc_model(input_values).logits
    labels = ctc_processor(text=text, return_tensors="pt").input_ids[0]

    emissions = torch.log_softmax(logits, dim=-1)[0].cpu()

    # TODO Attempting to smooth out 
    log_priors = torch.log(torch.bincount(labels, minlength=emissions.size(1)).float() + 1e-8)
    alpha = 0.5
    emissions = emissions - alpha * log_priors.unsqueeze(0)

    # 3. Generate Trellis
    blank_id = ctc_processor.tokenizer.pad_token_id
    token_path = [blank_id] + [val for t in labels for val in (t, blank_id)]
    
    trellis = torch.full((emissions.shape[0], len(token_path)), -float("inf"))
    trellis[0, 0] = emissions[0, blank_id]
    trellis[0, 1] = emissions[0, token_path[1]]

    for t in range(1, emissions.shape[0]):
        for j in range(len(token_path)):
            prev_trellis = trellis[t - 1, j]
            if j > 0: prev_trellis = max(prev_trellis, trellis[t-1, j-1])
            if j > 1 and token_path[j] != blank_id and token_path[j-2] == token_path[j]:
                prev_trellis = max(prev_trellis, trellis[t-1, j-2])
            trellis[t, j] = prev_trellis + emissions[t, token_path[j]]

    # 4. Backtrack
    path = []
    j = trellis.shape[1] - 1
    for t in range(trellis.shape[0] - 1, -1, -1):
        # Find the index of the maximum predecessor
        if j > 1 and token_path[j] != blank_id and token_path[j-2] == token_path[j] and trellis[t-1,j-2] >= trellis[t-1,j-1] and trellis[t-1,j-2] >= trellis[t-1,j]:
            j = j - 2
        elif j > 0 and trellis[t-1, j-1] >= trellis[t-1, j]:
            j = j - 1
        path.append((token_path[j], t))
    path.reverse()
    
    # 5. Merge and Format Segments
    segments = []
    for token, time_idx in path:
        if token != blank_id:
            char = ctc_processor.decode(token)
            if not segments or segments[-1]['char'] != char:
                segments.append({'char': char, 'start_frame': time_idx, 'end_frame': time_idx})
            else:
                segments[-1]['end_frame'] = time_idx
                    
    # segments = []
    # for i, p_idx in enumerate(path):
    #     if p_idx % 2 != 0: # Character token, not blank
    #         char = ctc_processor.decode(token_path[p_idx])
    #         if not segments or segments[-1]['char'] != char:
    #             segments.append({'char': char, 'start_frame': i, 'end_frame': i})
    #         else:
    #             segments[-1]['end_frame'] = i

    ratio = waveform.shape[1] / emissions.shape[0] / 16000
    print(ratio)
    return [{
        "word": seg['char'],
        "start": round(seg['start_frame'] * ratio, 3),
        "end": round((seg['end_frame'] + 1) * ratio, 3)
    } for seg in segments]


def align_audio_to_text(audio_path, transcription):
    """
    Performs forced alignment of an audio file to its transcription using a Wav2Vec2 model.

    Args:
        audio_path (str): Path to the audio file (e.g., .wav).
        transcription (str): The text transcription corresponding to the audio.

    Returns:
        list: A list containing a dictionary with the full segment and word-level (character-level for Chinese) timestamps.
    """
    # --- 1. Setup Model and Audio ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    
    # Load model and processor
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    
    # Load and resample audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # --- 2. Get Model Predictions ---
    # Process audio and text
    input_values = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values.to(device)
    labels = processor(text=transcription, return_tensors="pt").input_ids.to(device)

    # Get emission probabilities
    with torch.no_grad():
        logits = model(input_values).logits
    emissions = torch.log_softmax(logits, dim=-1)[0].cpu()
    
    # --- 3. Core Alignment Algorithm (CTC Forced Alignment) ---
    # Create the trellis (dynamic programming table)
    blank_id = processor.tokenizer.pad_token_id
    tokens = labels[0].tolist()
    
    # Insert blank tokens between characters for CTC
    token_path = [blank_id] + [val for t in tokens for val in (t, blank_id)]
    
    trellis = torch.full((emissions.shape[0], len(token_path)), -float("inf"))
    trellis[0, 0] = emissions[0, blank_id]
    trellis[0, 1] = emissions[0, token_path[1]]

    for t in range(1, emissions.shape[0]):
        for j in range(len(token_path)):
            # Case 1: Stay at the same token (can be blank or a character)
            prev_trellis = trellis[t - 1, j]
            # Case 2: Move from the previous token
            if j > 0:
                prev_trellis = max(prev_trellis, trellis[t-1, j-1])
            # Case 3: Skip a blank token
            if j > 1 and token_path[j] != blank_id and token_path[j-2] == token_path[j]:
                prev_trellis = max(prev_trellis, trellis[t-1, j-2])
            
            trellis[t, j] = prev_trellis + emissions[t, token_path[j]]

    # Backtrack to find the most likely path
    path = []
    j = trellis.shape[1] - 1
    for t in range(trellis.shape[0] - 1, -1, -1):
        # Find the index of the maximum predecessor in the trellis
        if j > 1 and token_path[j] != blank_id and token_path[j-2] == token_path[j] and trellis[t-1,j-2] >= trellis[t-1,j-1] and trellis[t-1,j-2] >= trellis[t-1,j]:
             path.append((token_path[j], t, emissions[t, token_path[j]].exp().item()))
             j = j-2
        elif j > 0 and trellis[t-1, j-1] >= trellis[t-1, j]:
            path.append((token_path[j], t, emissions[t, token_path[j]].exp().item()))
            j = j - 1
        else:
            path.append((token_path[j], t, emissions[t, token_path[j]].exp().item()))

    path.reverse()
    
    # --- 4. Merge Segments and Format Output ---
    # Filter out blank tokens and merge repeated characters
    char_segments = []
    for token, time_idx, score in path:
        if token != blank_id:
            char = processor.decode(token)
            if not char_segments or char_segments[-1]['char'] != char:
                char_segments.append({'char': char, 'start_frame': time_idx, 'end_frame': time_idx, 'scores': [score]})
            else:
                char_segments[-1]['end_frame'] = time_idx
                char_segments[-1]['scores'].append(score)
                
    # Convert frame indices to seconds
    ratio = waveform.shape[1] / emissions.shape[0]
    word_segs = []
    for seg in char_segments:
        word_segs.append({
            "word": seg['char'],
            "start": round(seg['start_frame'] * ratio, 3),
            "end": round((seg['end_frame'] + 1) * ratio, 3),
            "score": round(np.mean(seg['scores']), 3)
        })

    return [{
        "start": word_segs[0]["start"],
        "end": word_segs[-1]["end"],
        "text": transcription,
        "words": word_segs
    }]


def _format_mfa_transcript(text):
    """Whitespace-separated tokens; Mandarin MFA models expect segmented orthography."""
    return " ".join(ch for ch in text if not ch.isspace())


def _prepare_wav_for_mfa(audio_path):
    """16 kHz mono 16-bit PCM WAV (Kaldi/MFA requirement)."""
    y, _ = librosa.load(audio_path, sr=16000, mono=True)
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(out_path, y, 16000, subtype="PCM_16")
    return out_path


def _run_mfa(cmd):
    """Run MFA CLI; return CompletedProcess (check returncode yourself)."""
    return subprocess.run(cmd, capture_output=True, text=True)


def _mfa_error_message(cmd, result):
    detail = (result.stderr or result.stdout or "No output from MFA.").strip()
    return (
        "Montreal Forced Aligner failed.\n\n"
        f"Command: {' '.join(cmd)}\n\n"
        f"{detail[-6000:]}"
    )


def _parse_mfa_textgrid(tg_path):
    """Read MFA word intervals from a Praat TextGrid (praatio 6.x API)."""
    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
    tier_name = "words" if "words" in tg.tierNames else None
    if tier_name is None:
        for name in tg.tierNames:
            if "word" in name.lower():
                tier_name = name
                break
    if tier_name is None:
        return []
    words_tier = tg.getTier(tier_name)
    return [
        {"word": entry.label, "start": entry.start, "end": entry.end}
        for entry in words_tier.entries
        if entry.label.strip()
    ]


def mfa_align(audio_path, text):
    """Performs forced alignment using the Montreal Forced Aligner (MFA)."""
    if not shutil.which("mfa"):
        raise gr.Error(
            "Montreal Forced Aligner (`mfa`) is not installed or not on PATH. "
            "Activate your MFA conda env in the same shell as the app. See README.md."
        )

    mfa_text = _format_mfa_transcript(text)
    mfa_wav = _prepare_wav_for_mfa(audio_path)
    lab_path = tempfile.NamedTemporaryFile(delete=False, suffix=".lab", mode="w", encoding="utf-8")
    tg_path = tempfile.NamedTemporaryFile(delete=False, suffix=".TextGrid").name
    corpus_dir = None
    mfa_output_dir = None

    try:
        lab_path.write(mfa_text)
        lab_path.close()

        # --no_tokenization: we already pass space-separated characters (see _format_mfa_transcript)
        # Batch align first (more reliable for mandarin_mfa than align_one; see MFA issue #908)
        corpus_dir = tempfile.mkdtemp()
        mfa_output_dir = tempfile.mkdtemp()
        speaker_dir = os.path.join(corpus_dir, "speaker")
        os.makedirs(speaker_dir)
        shutil.copy(mfa_wav, os.path.join(speaker_dir, "utt1.wav"))
        with open(os.path.join(speaker_dir, "utt1.lab"), "w", encoding="utf-8") as f:
            f.write(mfa_text)

        align_cmd = [
            "mfa", "align", "--clean", "--single_speaker", "--no_tokenization", "-j", "1",
            corpus_dir, MFA_DICTIONARY, MFA_ACOUSTIC_MODEL, mfa_output_dir,
        ]
        align_result = _run_mfa(align_cmd)
        if align_result.returncode == 0:
            tg_path_batch = os.path.join(mfa_output_dir, "speaker", "utt1.TextGrid")
            if not os.path.exists(tg_path_batch):
                tg_path_batch = os.path.join(mfa_output_dir, "utt1.TextGrid")
            if os.path.exists(tg_path_batch):
                segments = _parse_mfa_textgrid(tg_path_batch)
                if segments:
                    return segments

        align_one_cmd = [
            "mfa", "align_one", "--clean", "--no_tokenization", "-j", "1",
            mfa_wav, lab_path.name, MFA_DICTIONARY, MFA_ACOUSTIC_MODEL, tg_path,
        ]
        align_one_result = _run_mfa(align_one_cmd)
        if align_one_result.returncode == 0 and os.path.exists(tg_path):
            segments = _parse_mfa_textgrid(tg_path)
            if segments:
                return segments

        failed = align_result if align_result.returncode != 0 else align_one_result
        failed_cmd = align_cmd if align_result.returncode != 0 else align_one_cmd
        raise gr.Error(_mfa_error_message(failed_cmd, failed))

    finally:
        for path in (mfa_wav, lab_path.name, tg_path):
            if path and os.path.exists(path):
                os.remove(path)
        if corpus_dir:
            shutil.rmtree(corpus_dir, ignore_errors=True)
        if mfa_output_dir:
            shutil.rmtree(mfa_output_dir, ignore_errors=True)

# --- Plotting Function ---

def create_pronunciation_plot(segments, audio_path):
    """Creates a Plotly figure showing pitch contour and aligned characters with tone colors."""
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    pitch_times = pitch.xs()

    fig = go.Figure()

    # Add pitch contour
    fig.add_trace(go.Scatter(
        x=pitch_times,
        y=pitch_values,
        mode='lines',
        line=dict(color='red', width=2),
        name='Pitch (F0)'
    ))
    
    # Add aligned segments and tone colors
    for i, seg in enumerate(segments):
        start_time, end_time, char = seg['start'], seg['end'], seg['word']
        print(start_time, end_time, char)
        
        # Get tone color
        tone_char = TARGET_PINYIN_TONES[i][0][-1]
        color = TONE_COLORS.get(tone_char, TONE_COLORS['5'])

        # Add colored background strip
        fig.add_shape(
            type="rect",
            x0=start_time, y0=0, x1=end_time, y1=1,
            xref="x", yref="paper",
            fillcolor=color,
            layer="below",
            line_width=0,
        )
        
        # Add character annotation at the top
        fig.add_annotation(
            x=(start_time + end_time) / 2,
            y=0.95,
            yref="paper",
            text=f"<b>{char}</b>",
            showarrow=False,
            font=dict(size=20, color="black")
        )

    # Update layout for a clean look
    fig.update_layout(
        title="Pronunciation Analysis: Pitch Contour and Alignment",
        xaxis_title="Time (s)",
        yaxis_title="Pitch (Hz)",
        showlegend=False,
        xaxis=dict(range=[0, snd.duration]),
        yaxis=dict(rangemode='tozero'), # Ensure y-axis starts at 0
        height=400,
    )

    return fig

# --- Main Analysis Function ---

def analyze_pronunciation(audio_filepath, alignment_method):
    if audio_filepath is None:
        raise gr.Error("Please record or upload audio first.")

    # 1. Pre-process audio
    trimmed_audio_path = trim_silence(audio_filepath)
    
    # 2. Recognize speech (for comparison)
    recognized_hanzi = recognize_speech(trimmed_audio_path)
    recognized_pinyin = format_pinyin(recognized_hanzi)
    
    # 3. Perform Forced Alignment using the selected method
    if "CTC" in alignment_method:
        # Using the target text for alignment is correct for a pronunciation practice app
        segments = ctc_align(trimmed_audio_path, TARGET_TEXT)
        # print(align_audio_to_text(trimmed_audio_path, TARGET_TEXT))
    elif "MFA" in alignment_method:
        segments = mfa_align(trimmed_audio_path, TARGET_TEXT)
    else:
        raise gr.Error("Invalid alignment method selected.")
        
    if not segments:
        raise gr.Error("Alignment failed. The audio might be too quiet, too noisy, or too different from the target text.")

    # 4. Create the plot
    fig = create_pronunciation_plot(segments, trimmed_audio_path)
    
    heard_html = phrase_card_html(
        "What we heard",
        recognized_hanzi.strip() if recognized_hanzi else "—",
        recognized_pinyin,
        variant="heard",
    )

    os.remove(trimmed_audio_path)

    return heard_html, fig

# --- Gradio UI ---

with gr.Blocks(css=PHRASE_CARD_CSS, title="Mandarin Speech Coach") as demo:
    gr.Markdown("# Mandarin Pronunciation Practice Tool")

    with gr.Row(elem_classes=["phrase-row"]):
        target_phrase = gr.HTML(value=TARGET_PHRASE_HTML)
        heard_phrase = gr.HTML(value=HEARD_PLACEHOLDER_HTML)

    with gr.Row():
        with gr.Column(scale=1):
            mic_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record Your Voice")
            alignment_method_dd = gr.Radio(
                ["CTC (Default)", "MFA"],
                label="Alignment Method",
                info="CTC is faster and runs anywhere. MFA is classic but requires local setup.",
                value="CTC (Default)"
            )
            analyze_btn = gr.Button("Analyze Pronunciation", variant="primary")

        with gr.Column(scale=2):
            output_plot = gr.Plot()

    analyze_btn.click(
        fn=analyze_pronunciation,
        inputs=[mic_input, alignment_method_dd],
        outputs=[heard_phrase, output_plot],
    )

demo.launch(
    debug=os.environ.get("GRADIO_DEBUG", "0") == "1",
    server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
    server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
)
