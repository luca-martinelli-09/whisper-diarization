import logging
from pathlib import Path
from typing import Tuple

import torch
from openai import OpenAI
from pyannote.audio import Pipeline
from pydub import AudioSegment

from src.settings import settings


def get_pipeline(filename: str | Path) -> Tuple[AudioSegment, Pipeline]:
    pipeline = Pipeline.from_pretrained(
        settings.PYANNOTE_MODEL,
        token=settings.HF_API_KEY,
    )
    pipeline.to(torch.device("cuda"))

    logging.info(f"🎙️ Processing {filename}")

    audio_segment = AudioSegment.from_mp3(filename)
    wav_audio = settings.tmp_folder.joinpath(filename).with_suffix(".wav")

    with open(wav_audio, "wb"):
        audio_segment.export(wav_audio, format="wav")

    return (audio_segment, pipeline(wav_audio))
