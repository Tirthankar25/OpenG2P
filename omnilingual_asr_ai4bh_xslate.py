# omnilingualASR — uses ai4bharat/indic-seamless for native Indic script ASR
# Fine-tuned SeamlessM4T specifically for Indian languages
# Correctly outputs native script (Bengali, Hindi, Kannada, Tamil, Gujarati, Marathi)

import torch
import warnings
warnings.filterwarnings('ignore')

import os
import time
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Successfully authenticated with HF_TOKEN.")
else:
    raise Exception("Error: HF_TOKEN environment variable not found.")

# ---------------------------------------------------------------------------
# Supported languages — SeamlessM4T language codes
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "hin": "Hindi",
    "eng": "English",
    "ben": "Bengali",
    "kan": "Kannada",
    "tam": "Tamil",
    "ory": "Odia",
    "guj": "Gujarati",   # confirmed supported: BhasaAnuvaad/indic-seamless covers 13 Indic languages incl. Gujarati/Marathi
    "mar": "Marathi",
}


class OmnilingualASR:
    """
    Native Indic script ASR using ai4bharat/indic-seamless.

    This is a fine-tuned version of SeamlessM4Tv2 specifically trained
    on Indian language data — unlike the base facebook/seamless-m4t-v2-large
    which defaults to English translation, this model correctly outputs
    native Indic scripts when tgt_lang is set to the source language.

    Usage pattern from official HuggingFace model card:
        processor = SeamlessM4TFeatureExtractor
        tokenizer = SeamlessM4TTokenizer
        text_out  = model.generate(**inputs, tgt_lang=lang)[0].numpy()
        text      = tokenizer.decode(text_out, skip_special_tokens=True)
    """

    def __init__(self, model_name="ai4bharat/indic-seamless"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print(f"Loading {model_name}…")
        from transformers import (
            SeamlessM4Tv2ForSpeechToText,
            SeamlessM4TTokenizer,
            SeamlessM4TFeatureExtractor,
        )

        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            model_name
        ).to(self.device)

        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = SeamlessM4TTokenizer.from_pretrained(model_name)

        print("Model ready.")

    def load_audio(self, audio_path, target_sr=16000):
        """Load audio using librosa — handles resampling + mono."""
        import librosa
        waveform, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        return torch.tensor(waveform)

    def transcribe(self, audio_path: str, src_lang: str, skip_translation: bool = False,
                   on_native_done=None) -> dict:
        """
        Transcribe audio in native script.

        Args:
            audio_path : WAV file path (16kHz mono)
            src_lang   : language code e.g. 'ben', 'hin', 'eng', 'tam', 'kan', 'ory'
            skip_translation : if True, skip the second (English) generate()
                pass entirely and return translation="". Used during the
                one-time voice enrollment, where only the native-script
                transcript is needed (paired with the reference audio) —
                running a whole extra English-translation pass on every
                enrollment utterance was pure wasted GPU/CPU time (and
                showed misleading English text while the person was just
                reading out loud to set up their voice, not talking to the
                assistant). See _transcribe_utterance() in the main app.
            on_native_done : optional callable(native_text: str), invoked
                right after the FIRST generate() pass (native script)
                finishes — before the second (English translation) pass
                even starts. Lets the caller push the native transcript to
                the browser immediately instead of waiting for both passes
                to complete, so the visible transcript shows up roughly as
                soon as it's ready rather than after the (often similarly
                slow) translation pass on top of it. Total per-utterance
                compute is unchanged — this only changes when the caller
                learns about the first half of it. Any exception the
                callback raises is caught and logged here, never allowed
                to break transcription itself. See _transcribe_utterance()
                in the main app for the caller side of this two-step push.
        """
        if src_lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{src_lang}'. "
                f"Choose from: {list(SUPPORTED_LANGUAGES.keys())}"
            )

        audio = self.load_audio(audio_path)

        # Feature extraction — note: positional arg (not audio= keyword)
        # as per official indic-seamless model card
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).to(self.device)

        print(f"[MODEL] src={src_lang} tgt={src_lang} device={self.device}")

        # 1. Native script transcription (tgt_lang = src_lang)
        _t0_native = time.time()
        with torch.no_grad():
            output_native = self.model.generate(
                **inputs,
                tgt_lang=src_lang,
            )
        _native_elapsed = time.time() - _t0_native
        text_native = output_native[0].cpu().numpy().squeeze()
        transcription = self.tokenizer.decode(
            text_native,
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        ).strip()

        if on_native_done is not None:
            try:
                on_native_done(transcription)
            except Exception as e:
                print(f"[MODEL] on_native_done callback error: {e}")

        # 2. English translation (tgt_lang = "eng") — skip if already English,
        #    or if the caller doesn't need it at all (skip_translation=True,
        #    e.g. during voice enrollment — see the docstring above).
        translation = ""
        _translation_elapsed = 0.0
        if skip_translation:
            translation = ""
        elif src_lang != "eng":
            _t0_translation = time.time()
            with torch.no_grad():
                output_eng = self.model.generate(
                    **inputs,
                    tgt_lang="eng",
                )
            _translation_elapsed = time.time() - _t0_translation
            text_eng = output_eng[0].cpu().numpy().squeeze()
            translation = self.tokenizer.decode(
                text_eng,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            ).strip()
        else:
            translation = transcription   # English input → same text

        # Split timing for the two generate() passes — printed separately
        # (rather than only one combined figure at the call site) so a
        # slow-transcription report can actually be pinned down: a large
        # _native_elapsed with device=cpu above means it's genuinely
        # running on CPU; both passes being individually fast but the
        # overall pipeline still feeling slow points instead at a queue
        # backlog (utterances arriving faster than one at a time can be
        # drained — see _utterance_worker() in the main app) rather than
        # per-call model latency.
        print(f"[MODEL] native_pass={_native_elapsed:.2f}s "
              f"translation_pass={_translation_elapsed:.2f}s "
              f"(skip_translation={skip_translation})")

        return {
            "transcription": transcription,   # native script
            "translation":   translation,     # English
            "used_language": src_lang,
            "language_name": SUPPORTED_LANGUAGES[src_lang],
        }
