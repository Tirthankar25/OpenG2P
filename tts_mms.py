# tts_mms.py — in-process Meta MMS-TTS (Massively Multilingual Speech):
# one small, FAST, non-autoregressive VITS model PER LANGUAGE, still no
# reference clip / voice cloning.
#

import io
import os
import subprocess

from tts_text_normalize import normalize_for_tts

SUPPORTED_LANGUAGES = [
    "hin", "eng", "ben", "kan", "tam", "ory", "guj", "mar",
]

# This app's session language codes -> the Hub id of that language's own
# MMS-TTS checkpoint. All 8 app languages have a dedicated MMS-TTS model.
_MODEL_IDS = {
    "hin": "facebook/mms-tts-hin",
    "eng": "facebook/mms-tts-eng",
    "ben": "facebook/mms-tts-ben",
    "kan": "facebook/mms-tts-kan",
    "tam": "facebook/mms-tts-tam",
    "ory": "facebook/mms-tts-ory",
    "guj": "facebook/mms-tts-guj",
    "mar": "facebook/mms-tts-mar",
}


def _uromanize(text: str, uroman_dir: str) -> str:
    """Shells out to uroman's own Perl script — see module docstring for
    why this is needed at all, and only for checkpoints that actually ask
    for it (tokenizer.is_uroman). Raises on any failure (missing $UROMAN,
    perl not installed, the script itself erroring) — synthesize() below
    catches this and turns it into a normal {"success": False, ...}
    result rather than letting it crash the caller."""
    script_path = os.path.join(uroman_dir, "bin", "uroman.pl")
    if not os.path.isfile(script_path):
        raise FileNotFoundError(
            f"uroman.pl not found under {uroman_dir!r} — this language's "
            f"TTS checkpoint needs uroman romanization. Clone it (`git "
            f"clone https://github.com/isi-nlp/uroman.git`) and set "
            f"UROMAN=<path to that clone> in the environment this app "
            f"runs in, then restart."
        )
    proc = subprocess.run(
        ["perl", script_path],
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"uroman.pl exited {proc.returncode}: "
                            f"{proc.stderr.decode('utf-8', errors='replace')}")
    return proc.stdout.decode("utf-8").rstrip("\n")


class MmsTTS:
    """Thin in-process wrapper around Meta's MMS-TTS: one (tokenizer,
    model) pair loaded per app language at construction (like
    OmnilingualASR/IndicTranslator/ParlerVoiceTTS before it); synthesize()
    is the only call site callers need. Same return-value contract as
    ParlerVoiceTTS.synthesize() (this module replaces) — {"success",
    "sampling_rate", "mime_type", "audio_bytes"} on success, {"success":
    False, "error"} on failure — so nothing calling into this needed to
    change shape, only the import/construction line.
    """

    def __init__(self, device: str = None):
        self.device = device
        self._voices: dict = {}   # lang_code -> {"tokenizer":…, "model":…, "uroman_dir": str|None}
        self._load_errors: dict = {}   # lang_code -> Exception, for languages that failed to load
        self._load()

    def _load(self):
        try:
            import torch
        except Exception as e:
            # No torch at all — nothing in this module can work; record
            # the same failure against every language so synthesize()'s
            # per-language error messages stay specific and actionable.
            for lang in SUPPORTED_LANGUAGES:
                self._load_errors[lang] = e
            print(f"[TTS] warning: could not import torch ({e}). The "
                  f"'speak reply' feature will be unavailable until this "
                  f"is fixed.")
            return

        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        uroman_dir = os.getenv("UROMAN")   # only consulted if a checkpoint asks for it

        from transformers import VitsModel, VitsTokenizer

        for lang, model_id in _MODEL_IDS.items():
            try:
                print(f"[TTS] loading {model_id} onto {self.device}…")
                tokenizer = VitsTokenizer.from_pretrained(model_id)
                model = VitsModel.from_pretrained(model_id).to(self.device)
                model.eval()
                if tokenizer.is_uroman and not uroman_dir:
                    # Don't fail the load outright — the checkpoint loaded
                    # fine, it's only CALLS to it that need uroman. Record
                    # it so synthesize() can give a precise error instead
                    # of a confusing crash deep inside tokenization.
                    print(f"[TTS] note: {model_id} needs uroman romanization "
                          f"(tokenizer.is_uroman=True) but the UROMAN env "
                          f"var isn't set — '{lang}' will report a clear "
                          f"error until that's configured (see "
                          f"tts_mms.py's module docstring).")
                self._voices[lang] = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "uroman_dir": uroman_dir if tokenizer.is_uroman else None,
                }
            except Exception as e:
                self._load_errors[lang] = e
                print(f"[TTS] warning: could not load {model_id} for "
                      f"'{lang}' ({e}). That language's 'speak reply' will "
                      f"be unavailable until this is fixed.")

        loaded = ", ".join(sorted(self._voices.keys())) or "(none)"
        print(f"[TTS] MMS-TTS ready for: {loaded}")

    def available(self) -> bool:
        """True if at least one language's model loaded — mirrors the
        startup-banner convention every other model in this app uses
        (print "ready" vs "unavailable"). A specific language missing is
        reported per-call by synthesize() below, not here."""
        return len(self._voices) > 0

    def synthesize(self, text: str, language: str) -> dict:
        """Synthesize `text` using `language`'s dedicated MMS-TTS model (a
        session's 3-letter code, e.g. one of SUPPORTED_LANGUAGES above —
        an unrecognized or unavailable code falls back to 'eng' if THAT
        loaded successfully, same fallback spirit as the old
        sooktam_language_for()/_voice_for() before it). No reference
        audio, no ref_text, no voice selection — one fixed voice per
        language. Never raises: always returns a dict with "success".

        On success: {"success": True, "sampling_rate": int,
                      "mime_type": "audio/wav", "audio_bytes": bytes}
        On failure: {"success": False, "error": str}
        """
        text = (text or "").strip()
        if not text:
            return {"success": False, "error": "No text to speak."}

        lang = (language or "").strip().lower()
        if lang not in self._voices:
            fallback = "eng"
            if fallback in self._voices and fallback != lang:
                lang = fallback
            elif lang not in self._load_errors:
                return {"success": False,
                        "error": f"No MMS-TTS model configured for language {language!r}."}

        if lang not in self._voices:
            err = self._load_errors.get(lang, "model failed to load")
            return {"success": False,
                    "error": f"TTS model unavailable for '{lang}' ({err})"}

        voice = self._voices[lang]
        try:
            import torch

            # Expand symbols the tokenizer's character vocab can't speak
            # (°, %, ₹, "10-12" style ranges, "hrs"/"mins") into plain
            # words BEFORE tokenizing — see tts_text_normalize.py's module
            # docstring for why MMS-TTS needs this at all (short version:
            # VITS has no numeric/symbol normalization front-end of its
            # own, so an out-of-vocab character is just silently dropped
            # rather than erroring, which is what was happening here).
            text = normalize_for_tts(text, lang)

            input_text = text
            if voice["uroman_dir"]:
                input_text = _uromanize(text, voice["uroman_dir"])

            tokenizer = voice["tokenizer"]
            model     = voice["model"]

            print(f"[TTS] synthesizing ({lang}, {len(text)} chars)…")
            inputs = tokenizer(text=input_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = model(**inputs)

            waveform = output.waveform[0].to(torch.float32).cpu().numpy()
            sr = model.config.sampling_rate

            import soundfile as sf
            out_buffer = io.BytesIO()
            sf.write(out_buffer, waveform, sr, format="WAV")
            return {
                "success": True,
                "sampling_rate": sr,
                "mime_type": "audio/wav",
                "audio_bytes": out_buffer.getvalue(),
            }
        except Exception as e:
            import traceback
            print(f"[TTS] synthesis failed: {e!r}")
            traceback.print_exc()
            return {"success": False, "error": f"Inference failure: {e}"}
