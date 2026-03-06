import os
import re

from dataclasses import dataclass
from typing import Callable


@dataclass
class Passage:
    text: str
    index: int
    label: str | None = None


class Document:
    """A document represented as an ordered list of passages with optional structural labels."""

    def __init__(self, passages: list[Passage], title: str = ""):
        self.passages = passages
        self.title = title

    def __repr__(self) -> str:
        return f"Document(title={self.title!r}, passages={len(self.passages)})"

    def __len__(self) -> int:
        return len(self.passages)

    def __getitem__(self, idx: int) -> Passage:
        return self.passages[idx]

    def __iter__(self):
        return iter(self.passages)

    def texts(self) -> list[str]:
        return [p.text for p in self.passages]

    def labels(self) -> list[str | None]:
        return [p.label for p in self.passages]

    @classmethod
    def from_file(
        cls,
        path: str,
        splitter: str | Callable = "line",
        preprocess_fn: Callable[[str], str] | None = None,
        title: str | None = None,
    ) -> 'Document':
        """Load a document from a file or directory.

        Args:
            path: Path to a .md file, plain text file, or directory.
            splitter: How to split content into passages. One of 'line', 'paragraph',
                      'section', 'sentence', or a callable that takes a string and
                      returns a list of strings.
            preprocess_fn: Optional function applied to each passage text after splitting.
            title: Document title. Inferred from filename/dirname if not provided.
        """
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            return cls._from_directory(path, splitter, preprocess_fn, title)
        ext = os.path.splitext(path)[1].lower()
        if ext == '.md':
            return cls._from_markdown(path, splitter, preprocess_fn, title)
        return cls._from_text(path, splitter, preprocess_fn, title)

    @classmethod
    def from_passages(
        cls,
        texts: list[str],
        labels: list[str] | None = None,
        title: str = "",
    ) -> 'Document':
        """Construct a Document directly from pre-split passage texts.

        Args:
            texts: List of passage strings.
            labels: Optional structural labels, one per passage.
            title: Document title.
        """
        if labels is not None and len(labels) != len(texts):
            raise ValueError("labels must have the same length as texts")
        passages = [
            Passage(text=t, index=i, label=labels[i] if labels else None)
            for i, t in enumerate(texts)
        ]
        return cls(passages, title=title)

    @classmethod
    def _from_markdown(cls, path, splitter, preprocess_fn, title) -> 'Document':
        with open(path, 'r') as f:
            lines = [line.rstrip('\n') for line in f]
        doc_title = title or os.path.splitext(os.path.basename(path))[0]
        passages = _parse_markdown(lines, splitter, preprocess_fn)
        for i, p in enumerate(passages):
            p.index = i
        return cls(passages, title=doc_title)

    @classmethod
    def _from_directory(cls, path, splitter, preprocess_fn, title) -> 'Document':
        doc_title = title or _path_to_title(path)
        passages = []
        _walk_directory(path, passages, splitter, preprocess_fn, label_parts=[])
        for i, p in enumerate(passages):
            p.index = i
        return cls(passages, title=doc_title)

    @classmethod
    def _from_text(cls, path, splitter, preprocess_fn, title) -> 'Document':
        with open(path, 'r') as f:
            content = f.read()
        doc_title = title or os.path.splitext(os.path.basename(path))[0]
        passages = []
        for t in _apply_splitter(content.splitlines(), splitter):
            if t.strip():
                text = preprocess_fn(t) if preprocess_fn else t
                passages.append(Passage(text=text, index=len(passages)))
        return cls(passages, title=doc_title)


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _parse_markdown(
    lines: list[str],
    splitter: str | Callable,
    preprocess_fn: Callable[[str], str] | None,
) -> list[Passage]:
    """Parse a markdown file into passages, tracking heading hierarchy as labels."""
    passages: list[Passage] = []
    heading_stack: list[tuple[int, str]] = []  # (depth, title)
    pending_lines: list[str] = []

    def current_label() -> str | None:
        if not heading_stack:
            return None
        return " > ".join(title for _, title in heading_stack)

    def flush():
        label = current_label()
        for t in _apply_splitter(pending_lines, splitter):
            if t.strip():
                text = preprocess_fn(t) if preprocess_fn else t
                passages.append(Passage(text=text, index=0, label=label))
        pending_lines.clear()

    for line in lines:
        m = re.match(r'^(#+)\s+(.*)', line)
        if m:
            flush()
            depth = len(m.group(1))
            heading_title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= depth:
                heading_stack.pop()
            heading_stack.append((depth, heading_title))
        else:
            pending_lines.append(line)

    flush()
    return passages


def _walk_directory(
    dirpath: str,
    passages: list[Passage],
    splitter: str | Callable,
    preprocess_fn: Callable[[str], str] | None,
    label_parts: list[str],
):
    """Recursively walk a directory, collecting passages with path-based labels.

    Markdown files are parsed for heading hierarchy, which is appended to the
    directory-path label. Plain text files receive a flat directory-path label.
    """
    for name in sorted(os.listdir(dirpath)):
        if name.startswith('.'):
            continue  # skip hidden files (.DS_Store, .gitignore, etc.)
        path = os.path.join(dirpath, name)
        title = _path_to_title(name)
        if os.path.isdir(path):
            _walk_directory(path, passages, splitter, preprocess_fn, label_parts + [title])
        elif os.path.isfile(path):
            label_prefix = " > ".join(label_parts + [title])
            if name.lower().endswith('.md'):
                with open(path, 'r') as f:
                    lines = [line.rstrip('\n') for line in f]
                file_passages = _parse_markdown(lines, splitter, preprocess_fn)
                for p in file_passages:
                    p.label = f"{label_prefix} > {p.label}" if p.label else label_prefix
                    passages.append(p)
            else:
                with open(path, 'r') as f:
                    content = f.read()
                for t in _apply_splitter(content.splitlines(), splitter):
                    if t.strip():
                        text = preprocess_fn(t) if preprocess_fn else t
                        passages.append(Passage(text=text, index=0, label=label_prefix))


def _apply_splitter(lines: list[str], splitter: str | Callable) -> list[str]:
    """Split a list of lines into passage strings according to the splitter strategy."""
    if callable(splitter):
        result = splitter("\n".join(lines))
        return result if isinstance(result, list) else [result]

    if splitter == "line":
        return [line.strip() for line in lines]

    if splitter == "paragraph":
        paragraphs, current = [], []
        for line in lines:
            if line.strip():
                current.append(line.strip())
            elif current:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))
        return paragraphs

    if splitter == "section":
        # All content under the current heading becomes one passage
        joined = " ".join(line.strip() for line in lines if line.strip())
        return [joined] if joined else []

    if splitter == "sentence":
        import nltk
        nltk.download('punkt_tab', quiet=True)
        text = " ".join(line.strip() for line in lines if line.strip())
        return nltk.sent_tokenize(text)

    raise ValueError(
        f"Unknown splitter: {splitter!r}. "
        "Choose from: 'line', 'paragraph', 'section', 'sentence', or a callable."
    )


def _path_to_title(path: str) -> str:
    """Convert a file/directory path to a human-readable title.

    Strips numeric sort prefixes (e.g. '00_Section_1.txt' -> 'Section 1').
    """
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    if name and name[0].isdigit():
        parts = name.split('_', 1)
        if len(parts) > 1:
            name = parts[1]
    return name.replace('_', ' ')
