import html
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class Transcript:
    video_id: str
    language: str
    language_code: str
    source: str
    segments: list[TranscriptSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "language": self.language,
            "language_code": self.language_code,
            "source": self.source,
            "segments": [asdict(segment) for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Transcript":
        segments = value.get("segments")
        if not isinstance(segments, list) or not segments or any(
            not isinstance(segment, dict) for segment in segments
        ):
            raise ValueError("Cached transcript has no valid segments")
        try:
            return normalize_snippets(
                video_id=str(value["video_id"]),
                snippets=segments,
                language=str(value.get("language", "Unknown")),
                language_code=str(value.get("language_code", "und")),
                source=str(value.get("source", "cache")),
            )
        except TranscriptUnavailable as exc:
            raise ValueError("Cached transcript has no usable segments") from exc


class TranscriptUnavailable(Exception):
    """No accessible transcript exists for this video and requested languages."""


def clean_text(value: str) -> str:
    value = html.unescape(TAG_RE.sub(" ", value))
    return SPACE_RE.sub(" ", value).strip()


def normalize_snippets(
    video_id: str,
    snippets: list[dict[str, Any]],
    language: str = "English",
    language_code: str = "en",
    source: str = "youtube",
) -> Transcript:
    """Normalize timestamped snippets; drop only exact, near-simultaneous duplicates."""
    if not VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError("Invalid YouTube video ID")
    segments: list[TranscriptSegment] = []
    for item in snippets:
        text = clean_text(str(item.get("text", "")))
        try:
            raw_start = float(item["start"])
            raw_duration = float(item.get("duration", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Transcript snippet has invalid timestamp metadata") from exc
        if not math.isfinite(raw_start) or not math.isfinite(raw_duration):
            raise ValueError("Transcript snippet timestamps must be finite numbers")
        start = max(0.0, raw_start)
        duration = max(0.0, raw_duration)
        if not text:
            continue
        if segments and text.casefold() == segments[-1].text.casefold() and start - segments[-1].start < 1.0:
            continue
        segments.append(TranscriptSegment(text=text, start=start, duration=duration))
    if not segments:
        raise TranscriptUnavailable("The video transcript was empty.")
    return Transcript(video_id, language, language_code, source, segments)


class YouTubeTranscriptLoader:
    """Thin adapter around youtube-transcript-api, injectable for deterministic tests."""

    def __init__(self, api_factory: Any = YouTubeTranscriptApi):
        self.api_factory = api_factory

    def fetch(self, video_id: str, languages: list[str]) -> Transcript:
        if not VIDEO_ID_RE.fullmatch(video_id):
            raise ValueError("Invalid YouTube video ID")
        try:
            fetched = self.api_factory().fetch(video_id, languages=languages)
        except Exception as exc:
            # Library exceptions vary by upstream version; do not manufacture transcript text.
            message = str(exc).strip() or exc.__class__.__name__
            raise TranscriptUnavailable(message[:300]) from exc

        raw = fetched.to_raw_data()
        source = "youtube_auto" if bool(getattr(fetched, "is_generated", False)) else "youtube_manual"
        return normalize_snippets(
            video_id=video_id,
            snippets=raw,
            language=str(getattr(fetched, "language", "Unknown")),
            language_code=str(getattr(fetched, "language_code", languages[0] if languages else "und")),
            source=source,
        )
