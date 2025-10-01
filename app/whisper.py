import logging
from pathlib import Path
from typing import List

from openai import OpenAI
from openai.types.audio import TranscriptionVerbose
from pyannote.pipeline import Pipeline
from pydantic import BaseModel
from pydub import AudioSegment

from app.settings import settings


class TranscriptSegment(BaseModel):
    audio_file: str | Path
    speaker: str
    i: str
    start: float
    end: float
    transcript: TranscriptionVerbose


def get_transcripts(
    diarization: Pipeline, audio_segment: AudioSegment
) -> List[TranscriptSegment]:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    transcripts = []

    for turn, i, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        logging.info(
            f"🧩 START={turn.start:.1f}s STOP={turn.end:.1f}s SPEAKER={speaker}"
        )

        start = turn.start * 1000
        end = turn.end * 1000

        chunck = audio_segment[slice(start, end)]

        chunk_filename = settings.tmp_folder.joinpath(f"segment-{start}.mp3")

        logging.info(f"💾 Saving chunck {chunk_filename}")

        with open(chunk_filename, "wb") as f:
            chunck.export(chunk_filename, format="mp3")

        audio_chunk_segment = open(chunk_filename, "rb")

        logging.info(f"🎙️ Processing {chunk_filename}")

        try:
            transcript = client.audio.transcriptions.create(
                file=audio_chunk_segment,
                model=settings.OPENAI_WHISPER_MODEL,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                prompt=settings.OPENAI_WHISPER_PROMPT,
            )

            logging.info(f" 📝 Got transcription for {chunk_filename}")

            transcripts.append(
                TranscriptSegment(
                    audio_file=chunk_filename,
                    speaker=speaker,
                    i=i,
                    start=start,
                    end=end,
                    transcript=transcript,
                )
            )
        except Exception as e:
            logging.error(e)

    return transcripts
