"""Load .md/.txt corpus files and apply light text cleanup.

PDF, HTML, and DOCX parsers are added in a later ingest step.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt"}
SKIP_NAMES = {"README.md", "README.txt"}
DEFAULT_CORPUS_DIR = Path("data/corpus")


def parse_document(path: Path) -> str:
    """Read a markdown or plain-text file as UTF-8.

    Args:
        path: File under the corpus directory.

    Returns:
        Raw file contents.

    Raises:
        ValueError: If the suffix is not .md or .txt.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        logger.debug("parse skip: unsupported suffix %s for %s", suffix, path.name)
        raise ValueError(f"Unsupported file type for A2 parse: {suffix}")
    text = path.read_text(encoding="utf-8")
    logger.debug("parse %s: %s chars", path.name, len(text))
    return text


def clean_text(text: str) -> str:
    """Normalise newlines and collapse extra blank lines.

    Does not change wording. Later steps may add PII scrubbing.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalised.split("\n")]
    collapsed: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                collapsed.append("")
        else:
            blank_run = 0
            collapsed.append(line)
    cleaned = "\n".join(collapsed).strip()
    logger.debug("clean_text: %s chars in → %s chars out", len(text), len(cleaned))
    return cleaned


def load_corpus(corpus_dir: Path | None = None) -> list[dict[str, str]]:
    """Load supported documents from corpus_dir.

    Skips README and unknown extensions. Each item has source_id (filename
    stem), path, and cleaned text.

    Args:
        corpus_dir: Folder of documents. Defaults to data/corpus.

    Returns:
        Sorted list of document dicts.
    """
    root = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS_DIR
    logger.debug("load_corpus dir=%s", root)
    documents: list[dict[str, str]] = []
    if not root.is_dir():
        logger.debug("load_corpus: not a directory, returning empty list")
        return documents
    for path in sorted(root.iterdir()):
        if not path.is_file():
            logger.debug("skip not a file: %s", path)
            continue
        if path.name in SKIP_NAMES:
            logger.debug("skip README: %s", path.name)
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            logger.debug("skip unsupported type: %s", path.name)
            continue
        text = clean_text(parse_document(path))
        logger.debug(
            "loaded source_id=%s chars=%s preview=%r",
            path.stem,
            len(text),
            text[:80].replace("\n", " "),
        )
        documents.append(
            {
                "source_id": path.stem,
                "path": str(path),
                "text": text,
            }
        )
    logger.debug("load_corpus done: %s documents", len(documents))
    return documents
