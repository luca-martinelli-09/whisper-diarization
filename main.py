import argparse
import logging
import shutil
from pathlib import Path

from app.diarization import get_pipeline
from app.settings import settings
from app.vtt import create_vtt
from app.whisper import get_transcripts

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    prog="Digital People - Whisper Diarization",
)
parser.add_argument("filename")
parser.print_help()

args = parser.parse_args()

settings.tmp_folder.mkdir(exist_ok=True)

if __name__ == "__main__":
    filename = Path(args.filename).absolute()

    if not filename.exists():
        logger.error(f"🛑 File {filename} not exists")
        exit(1)

    (audio_segment, diarization) = get_pipeline(args.filename)

    transcripts = get_transcripts(diarization, audio_segment)

    vtt = create_vtt(transcripts)

    with open(filename.with_suffix(".vtt"), "w") as f:
        vtt.write(f)

    shutil.rmtree(settings.tmp_folder.absolute())
