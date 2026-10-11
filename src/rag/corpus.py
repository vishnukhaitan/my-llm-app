"""Load .md/.txt corpus files and apply light text cleanup.

PDF, HTML, and DOCX parsers are added in a later ingest step.
"""

from __future__ import annotations

from pathlib import Path

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
        raise ValueError(f"Unsupported file type for A2 parse: {suffix}")
    return path.read_text(encoding="utf-8")


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
    return "\n".join(collapsed).strip()


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
    documents: list[dict[str, str]] = []
    if not root.is_dir():
        return documents
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES:
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = clean_text(parse_document(path))
        documents.append(
            {
                "source_id": path.stem,
                "path": str(path),
                "text": text,
            }
        )
    return documents
