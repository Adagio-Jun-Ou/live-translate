import logging
import threading
from typing import Optional, Callable

from .config import Config

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()

_MODEL_MAP = {
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
    ("ko", "en"): "Helsinki-NLP/opus-mt-ko-en",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
    ("pt", "en"): "Helsinki-NLP/opus-mt-pt-en",
    ("nl", "en"): "Helsinki-NLP/opus-mt-nl-en",
    ("tr", "en"): "Helsinki-NLP/opus-mt-tr-en",
    ("fa", "en"): "Helsinki-NLP/opus-mt-fa-en",
    ("ur", "en"): "Helsinki-NLP/opus-mt-ur-en",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("pl", "en"): "Helsinki-NLP/opus-mt-pl-en",
    ("uk", "en"): "Helsinki-NLP/opus-mt-uk-en",
    ("vi", "en"): "Helsinki-NLP/opus-mt-vi-en",
    ("he", "en"): "Helsinki-NLP/opus-mt-he-en",
    ("sv", "en"): "Helsinki-NLP/opus-mt-sv-en",
    ("da", "en"): "Helsinki-NLP/opus-mt-da-en",
    ("fi", "en"): "Helsinki-NLP/opus-mt-fi-en",
    ("nb", "en"): "Helsinki-NLP/opus-mt-nb-en",
    ("ro", "en"): "Helsinki-NLP/opus-mt-ro-en",
    ("bg", "en"): "Helsinki-NLP/opus-mt-bg-en",
    ("el", "en"): "Helsinki-NLP/opus-mt-el-en",
    ("hu", "en"): "Helsinki-NLP/opus-mt-hu-en",
    ("cs", "en"): "Helsinki-NLP/opus-mt-cs-en",
    ("sk", "en"): "Helsinki-NLP/opus-mt-sk-en",
    ("lt", "en"): "Helsinki-NLP/opus-mt-lt-en",
    ("lv", "en"): "Helsinki-NLP/opus-mt-lv-en",
    ("et", "en"): "Helsinki-NLP/opus-mt-et-en",
    ("id", "en"): "Helsinki-NLP/opus-mt-id-en",
    ("th", "en"): "Helsinki-NLP/opus-mt-th-en",
    ("ca", "en"): "Helsinki-NLP/opus-mt-ca-en",
    ("gl", "en"): "Helsinki-NLP/opus-mt-gl-en",
    ("cy", "en"): "Helsinki-NLP/opus-mt-cy-en",
    ("ga", "en"): "Helsinki-NLP/opus-mt-ga-en",
    ("mt", "en"): "Helsinki-NLP/opus-mt-mt-en",
    ("mk", "en"): "Helsinki-NLP/opus-mt-mk-en",
    ("sq", "en"): "Helsinki-NLP/opus-mt-sq-en",
    ("sr", "en"): "Helsinki-NLP/opus-mt-sr-en",
    ("hr", "en"): "Helsinki-NLP/opus-mt-hr-en",
    ("sl", "en"): "Helsinki-NLP/opus-mt-sl-en",
    ("is", "en"): "Helsinki-NLP/opus-mt-is-en",
    ("eu", "en"): "Helsinki-NLP/opus-mt-eu-en",
    ("eo", "en"): "Helsinki-NLP/opus-mt-eo-en",
    ("sw", "en"): "Helsinki-NLP/opus-mt-sw-en",
    ("tl", "en"): "Helsinki-NLP/opus-mt-tl-en",
    ("ms", "en"): "Helsinki-NLP/opus-mt-ms-en",
    ("mg", "en"): "Helsinki-NLP/opus-mt-mg-en",
    ("ml", "en"): "Helsinki-NLP/opus-mt-fr-en",
}


class Translator:
    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._ready = threading.Event()
        self._source_lang = config.source_language
        self._target_lang = config.target_language
        self._model_name = self._resolve_model(self._source_lang, self._target_lang)

    def _resolve_model(self, src: str, tgt: str) -> str:
        key = (src, tgt)
        if key in _MODEL_MAP:
            return _MODEL_MAP[key]
        return self.config.translation_model

    def _load_model(self):
        with _CACHE_LOCK:
            cache_key = self._model_name
            if cache_key in _MODEL_CACHE:
                self._model, self._tokenizer = _MODEL_CACHE[cache_key]
                self._ready.set()
                return

        try:
            from transformers import MarianMTModel, MarianTokenizer

            logger.info("Loading translation model '%s'...", self._model_name)

            self._tokenizer = MarianTokenizer.from_pretrained(
                self._model_name, cache_dir=self.config.cache_dir
            )
            self._model = MarianMTModel.from_pretrained(
                self._model_name, cache_dir=self.config.cache_dir
            )

            with _CACHE_LOCK:
                _MODEL_CACHE[cache_key] = (self._model, self._tokenizer)

            logger.info("Translation model loaded successfully")
            self._ready.set()
        except Exception as e:
            logger.error("Failed to load translation model: %s", e)
            raise

    def start(self):
        load_thread = threading.Thread(target=self._load_model, daemon=True)
        load_thread.start()

    def stop(self):
        pass

    def translate(self, text: str, src_lang: Optional[str] = None) -> str:
        if not self._ready.wait(timeout=5.0):
            return text

        if not text or not text.strip():
            return text

        try:
            inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            outputs = self._model.generate(
                **inputs,
                max_length=512,
                num_beams=1,
                early_stopping=False,
            )

            translated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated
        except Exception as e:
            logger.error("Translation error: %s", e)
            return text

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def supported_pairs() -> list:
        return [(s, t) for (s, t) in _MODEL_MAP.keys()]
