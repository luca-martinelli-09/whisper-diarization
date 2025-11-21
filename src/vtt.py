from typing import List

from webvtt import Caption, WebVTT

from src.whisper import TranscriptSegment


def format_milliseconds(milliseconds):
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"


def create_vtt(transcripts: List[TranscriptSegment]) -> WebVTT:
    vtt = WebVTT()

    for transcript in transcripts:
        for x in transcript.transcript.segments:
            start = transcript.start + x.start * 1000
            end = transcript.start + x.end * 1000

            caption = Caption(
                format_milliseconds(start),
                format_milliseconds(end),
                f"<v {transcript.speaker}>" + x.text,
            )

            vtt.captions.append(caption)

    return vtt
