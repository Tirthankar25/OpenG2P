# indic_translator.py — English -> Indic text translation using AI4Bharat's
# IndicTrans2, run as its own small dedicated model.
#

# IndicTrans2 is AI4Bharat's own purpose-built EN<->Indic MT model, so this
# keeps the whole pipeline in the same model family while actually working.
#
# --- Prereqs -------------------------------------------------------------
#   pip install indictranstoolkit sentencepiece
# ---------------------------------------------------------------------------

import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --- Compatibility shim ----------------------------------------------------

try:
    import transformers.tokenization_utils as _ttu
    if not hasattr(_ttu, "PreTrainedTokenizerBase"):
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
        _ttu.PreTrainedTokenizerBase = _PTB
except ImportError:
    pass   # let the real error surface below if transformers is broken some other way
# ---------------------------------------------------------------------------

try:
    from IndicTransToolkit import IndicProcessor
except ImportError as e:
    raise ImportError(
        "Could not import IndicProcessor from IndicTransToolkit "
        ) from e

# "dist-200M" is the distilled, lightweight checkpoint — much smaller
# footprint to sit alongside the ASR model and Llama on the same GPU.
# Swap for "ai4bharat/indictrans2-en-indic-1B" if you want higher quality
# and have the VRAM to spare.
INDICTRANS_MODEL = os.getenv("INDICTRANS_MODEL", "ai4bharat/indictrans2-en-indic-dist-200M")

# language codes used elsewhere in this app -> IndicTrans2's BCP47-ish codes
LANG_CODE_MAP = {
    "hin": "hin_Deva",
    "ben": "ben_Beng",
    "kan": "kan_Knda",
    "tam": "tam_Taml",
    "ory": "ory_Orya",
    "guj": "guj_Gujr",
    "mar": "mar_Deva",
    "eng": "eng_Latn",
}

SRC_LANG = "eng_Latn"   # this class only ever translates FROM English


class IndicTranslator:
    """English -> Indic text translation via ai4bharat/indictrans2-en-indic-*."""

    def __init__(self, model_name: str = INDICTRANS_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[MT] loading {model_name} on {self.device}…")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        self.ip = IndicProcessor(inference=True)
        print("[MT] IndicTrans2 ready.")

    def translate(self, text: str, target_lang: str) -> str:
        """Translate English `text` into `target_lang` (a LANG_CODE_MAP key,
        e.g. "hin", "ben", "kan", "tam", "ory").

        Returns `text` unchanged if target_lang is English, unrecognized, or
        text is empty — there's nothing to translate in those cases.
        """
        text = (text or "").strip()
        if not text:
            return text

        tgt_code = LANG_CODE_MAP.get(target_lang)
        if tgt_code is None or tgt_code == SRC_LANG:
            return text

        batch = self.ip.preprocess_batch([text], src_lang=SRC_LANG, tgt_lang=tgt_code)
        inputs = self.tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

    
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                use_cache=False,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        decoded = self.tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        translated = self.ip.postprocess_batch(decoded, lang=tgt_code)
        return translated[0].strip() if translated else ""
