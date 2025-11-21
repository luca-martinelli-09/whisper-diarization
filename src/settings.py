from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str
    OPENAI_WHISPER_MODEL: str
    OPENAI_WHISPER_PROMPT: str
    
    HF_API_KEY: str

    PYANNOTE_MODEL: str

    tmp_folder: Path = Path("tmp")


settings = Settings()
