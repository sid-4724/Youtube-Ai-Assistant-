from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.transcripts import Transcript


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def transcript_to_documents(
    transcript: Transcript,
    video_id: str,
    chunk_size_tokens: int = 320,
    chunk_overlap_tokens: int = 64,
) -> list[Document]:
    """Token-aware recursive chunks whose time metadata maps to covered source snippets."""
    if not transcript.segments:
        return []

    # The MiniLM default accepts 384 wordpieces. A 320-token window leaves room for
    # special tokens; 64-token overlap preserves short definitions across boundaries.
    parts: list[str] = []
    spans: list[tuple[int, int, float, float]] = []
    cursor = 0
    for segment in transcript.segments:
        text = segment.text.strip()
        if not text:
            continue
        if parts:
            cursor += 1  # newline separator
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((start, cursor, segment.start, segment.end))

    full_text = "\n".join(parts)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    raw_docs = splitter.create_documents([full_text])
    docs: list[Document] = []
    for index, document in enumerate(raw_docs):
        start_index = int(document.metadata.get("start_index", 0))
        end_index = start_index + len(document.page_content)
        covered = [span for span in spans if span[0] < end_index and span[1] > start_index]
        if not covered:
            continue
        start_seconds = min(span[2] for span in covered)
        end_seconds = max(span[3] for span in covered)
        docs.append(
            Document(
                page_content=document.page_content,
                metadata={
                    "video_id": video_id,
                    "chunk_id": f"{video_id}-{index:05d}",
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "start_timestamp": _format_timestamp(start_seconds),
                    "end_timestamp": _format_timestamp(end_seconds),
                    "language": transcript.language_code,
                    "transcript_source": transcript.source,
                },
            )
        )
    return docs
