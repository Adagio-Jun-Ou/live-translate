import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    whisper_model: str = "tiny"
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"
    whisper_num_workers: int = 2
    whisper_beam_size: int = 1
    whisper_vad_filter: bool = True
    whisper_language: Optional[str] = "ar"

    source_language: str = "ar"
    target_language: str = "en"
    translation_model: str = "Helsinki-NLP/opus-mt-ar-en"

    sample_rate: int = 16000
    chunk_duration: float = 0.5
    buffer_duration: float = 2.0
    max_buffer_duration: float = 15.0
    silence_threshold: float = 0.02
    silence_padding_ms: int = 400
    min_speech_duration: float = 0.3
    max_inactivity: float = 3.0

    audio_source: Optional[str] = None
    audio_device_index: Optional[int] = None
    use_system_audio: bool = True

    ws_host: str = "127.0.0.1"
    ws_port: int = 8765
    http_host: str = "127.0.0.1"
    http_port: int = 8766

    max_subtitle_length: int = 120
    max_history: int = 15

    log_level: str = "INFO"

    data_dir: str = field(default_factory=lambda: os.path.expanduser("~/.local/share/live-translate"))
    cache_dir: str = field(default_factory=lambda: os.path.join(os.path.expanduser("~/.cache"), "live-translate"))

    def ensure_dirs(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for field_name in cls.__dataclass_fields__:
            env_key = f"LT_{field_name.upper()}"
            val = os.environ.get(env_key)
            if val is not None:
                field_type = cls.__dataclass_fields__[field_name].type
                if field_type == bool:
                    setattr(cfg, field_name, val.lower() in ("1", "true", "yes"))
                elif field_type in (int, Optional[int]):
                    setattr(cfg, field_name, int(val))
                elif field_type == float:
                    setattr(cfg, field_name, float(val))
                elif field_type in (str, Optional[str]):
                    setattr(cfg, field_name, val)
        return cfg
