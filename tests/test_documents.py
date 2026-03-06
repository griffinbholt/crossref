"""Tests for crossref.documents."""
import csv
import json
import os

import pytest

from crossref.documents import (
    Document, Passage,
    _apply_splitter, _parse_markdown, _path_to_title,
)


# ---------------------------------------------------------------------------
# Passage
# ---------------------------------------------------------------------------

class TestPassage:
    def test_fields(self):
        p = Passage(text="hello world", index=0)
        assert p.text == "hello world"
        assert p.index == 0
        assert p.label is None

    def test_with_label(self):
        p = Passage(text="text", index=5, label="Book > Chapter > Verse")
        assert p.label == "Book > Chapter > Verse"


# ---------------------------------------------------------------------------
# Document basics
# ---------------------------------------------------------------------------

class TestDocumentBasics:
    def test_len(self, doc_a):
        assert len(doc_a) == 3

    def test_getitem(self, doc_a):
        assert doc_a[0].text == "the quick brown fox"

    def test_iter(self, doc_a):
        texts = [p.text for p in doc_a]
        assert texts == ["the quick brown fox", "jumps over the lazy dog", "hello world"]

    def test_texts(self, doc_a):
        assert doc_a.texts() == ["the quick brown fox", "jumps over the lazy dog", "hello world"]

    def test_labels(self, doc_a):
        assert doc_a.labels() == ["a:1", "a:2", "a:3"]

    def test_labels_none(self):
        doc = Document.from_passages(["a", "b"])
        assert doc.labels() == [None, None]

    def test_repr(self, doc_a):
        r = repr(doc_a)
        assert "Doc A" in r
        assert "3" in r

    def test_title(self, doc_a):
        assert doc_a.title == "Doc A"


# ---------------------------------------------------------------------------
# from_passages
# ---------------------------------------------------------------------------

class TestFromPassages:
    def test_basic(self):
        doc = Document.from_passages(["a", "b", "c"])
        assert len(doc) == 3
        assert doc[0].text == "a"
        assert doc[1].index == 1
        assert doc[0].label is None

    def test_with_labels(self):
        doc = Document.from_passages(["x", "y"], labels=["L1", "L2"], title="Test")
        assert doc[0].label == "L1"
        assert doc[1].label == "L2"
        assert doc.title == "Test"

    def test_mismatched_labels_raises(self):
        with pytest.raises(ValueError, match="same length"):
            Document.from_passages(["a", "b"], labels=["only one"])

    def test_indices_sequential(self):
        doc = Document.from_passages(["p", "q", "r"])
        assert [p.index for p in doc] == [0, 1, 2]


# ---------------------------------------------------------------------------
# from_file — text
# ---------------------------------------------------------------------------

class TestFromFileText:
    def test_plain_text_lines(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("line one\nline two\nline three\n")
        doc = Document.from_file(str(f))
        assert len(doc) == 3
        assert doc[0].text == "line one"

    def test_title_from_filename(self, tmp_path):
        f = tmp_path / "my_document.txt"
        f.write_text("hello\n")
        doc = Document.from_file(str(f))
        assert doc.title == "my_document"

    def test_title_override(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello\n")
        doc = Document.from_file(str(f), title="My Custom Title")
        assert doc.title == "My Custom Title"

    def test_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("line one\n\n\nline two\n")
        doc = Document.from_file(str(f))
        assert len(doc) == 2

    def test_data_extension_raises(self, tmp_path):
        for ext in (".csv", ".tsv", ".json", ".jsonl", ".ndjson"):
            f = tmp_path / f"file{ext}"
            f.write_text("data")
            with pytest.raises(ValueError, match="from_csv|from_json"):
                Document.from_file(str(f))

    def test_paragraph_splitter(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("line one\nline two\n\nline three\nline four\n")
        doc = Document.from_file(str(f), splitter="paragraph")
        assert len(doc) == 2
        assert "line one" in doc[0].text
        assert "line two" in doc[0].text


# ---------------------------------------------------------------------------
# from_file — markdown
# ---------------------------------------------------------------------------

class TestFromFileMarkdown:
    def test_markdown_labels(self, tmp_path):
        f = tmp_path / "book.md"
        f.write_text(
            "# Genesis\n"
            "## Chapter 1\n"
            "In the beginning.\n"
            "## Chapter 2\n"
            "And the earth.\n"
        )
        doc = Document.from_file(str(f))
        labels = doc.labels()
        assert any("Genesis" in (l or "") and "Chapter 1" in (l or "") for l in labels)
        assert any("Genesis" in (l or "") and "Chapter 2" in (l or "") for l in labels)

    def test_markdown_sequential_indices(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# H\nline 1\nline 2\n")
        doc = Document.from_file(str(f))
        assert [p.index for p in doc] == list(range(len(doc)))

    def test_markdown_title_from_filename(self, tmp_path):
        f = tmp_path / "my_book.md"
        f.write_text("# Chapter\ntext\n")
        doc = Document.from_file(str(f))
        assert doc.title == "my_book"


# ---------------------------------------------------------------------------
# from_file — directory
# ---------------------------------------------------------------------------

class TestFromDirectory:
    def test_directory_load(self, tmp_path):
        (tmp_path / "a.txt").write_text("passage one\npassage two\n")
        (tmp_path / "b.txt").write_text("passage three\n")
        doc = Document.from_file(str(tmp_path))
        assert len(doc) == 3

    def test_directory_labels_include_filename(self, tmp_path):
        (tmp_path / "section.txt").write_text("hello\n")
        doc = Document.from_file(str(tmp_path))
        assert any("section" in (p.label or "").lower() for p in doc)

    def test_directory_hidden_files_skipped(self, tmp_path):
        (tmp_path / ".hidden").write_text("secret\n")
        (tmp_path / "visible.txt").write_text("visible\n")
        doc = Document.from_file(str(tmp_path))
        assert len(doc) == 1

    def test_directory_data_extensions_skipped(self, tmp_path):
        (tmp_path / "data.csv").write_text("col\nval\n")
        (tmp_path / "real.txt").write_text("text\n")
        doc = Document.from_file(str(tmp_path))
        assert len(doc) == 1

    def test_directory_sequential_indices(self, tmp_path):
        (tmp_path / "a.txt").write_text("a\nb\n")
        (tmp_path / "b.txt").write_text("c\n")
        doc = Document.from_file(str(tmp_path))
        assert [p.index for p in doc] == list(range(len(doc)))


# ---------------------------------------------------------------------------
# from_csv
# ---------------------------------------------------------------------------

class TestFromCSV:
    def _write_csv(self, path, rows, delimiter=","):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)

    def test_basic(self, tmp_path):
        p = tmp_path / "data.csv"
        self._write_csv(str(p), [{"text": "hello"}, {"text": "world"}])
        doc = Document.from_csv(str(p), text_column="text")
        assert len(doc) == 2
        assert doc[0].text == "hello"

    def test_label_column(self, tmp_path):
        p = tmp_path / "data.csv"
        self._write_csv(str(p), [
            {"text": "hello", "ref": "1:1"},
            {"text": "world", "ref": "1:2"},
        ])
        doc = Document.from_csv(str(p), text_column="text", label_column="ref")
        assert doc[0].label == "1:1"
        assert doc[1].label == "1:2"

    def test_tsv_auto_delimiter(self, tmp_path):
        p = tmp_path / "data.tsv"
        self._write_csv(str(p), [{"text": "tab separated", "ref": "1"}], delimiter="\t")
        doc = Document.from_csv(str(p), text_column="text", label_column="ref")
        assert doc[0].text == "tab separated"

    def test_empty_text_rows_skipped(self, tmp_path):
        p = tmp_path / "data.csv"
        self._write_csv(str(p), [{"text": "hello"}, {"text": ""}])
        doc = Document.from_csv(str(p), text_column="text")
        assert len(doc) == 1

    def test_title_from_filename(self, tmp_path):
        p = tmp_path / "scripture.csv"
        self._write_csv(str(p), [{"text": "verse"}])
        doc = Document.from_csv(str(p), text_column="text")
        assert doc.title == "scripture"

    def test_title_override(self, tmp_path):
        p = tmp_path / "data.csv"
        self._write_csv(str(p), [{"text": "verse"}])
        doc = Document.from_csv(str(p), text_column="text", title="My Title")
        assert doc.title == "My Title"

    def test_sequential_indices(self, tmp_path):
        p = tmp_path / "data.csv"
        self._write_csv(str(p), [{"text": "a"}, {"text": "b"}, {"text": "c"}])
        doc = Document.from_csv(str(p), text_column="text")
        assert [p.index for p in doc] == [0, 1, 2]


# ---------------------------------------------------------------------------
# from_json
# ---------------------------------------------------------------------------

class TestFromJSON:
    def test_array(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps([{"verse": "In the beginning"}, {"verse": "And the earth"}]))
        doc = Document.from_json(str(p), text_key="verse")
        assert len(doc) == 2
        assert doc[0].text == "In the beginning"

    def test_dict_wrapping(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps({"verses": [{"text": "hello"}, {"text": "world"}]}))
        doc = Document.from_json(str(p), text_key="text")
        assert len(doc) == 2

    def test_label_key(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps([{"text": "hi", "ref": "1:1"}]))
        doc = Document.from_json(str(p), text_key="text", label_key="ref")
        assert doc[0].label == "1:1"

    def test_empty_text_skipped(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps([{"text": "hi"}, {"text": ""}]))
        doc = Document.from_json(str(p), text_key="text")
        assert len(doc) == 1

    def test_title_from_filename(self, tmp_path):
        p = tmp_path / "my_book.json"
        p.write_text(json.dumps([{"text": "verse"}]))
        doc = Document.from_json(str(p), text_key="text")
        assert doc.title == "my_book"


# ---------------------------------------------------------------------------
# from_jsonl
# ---------------------------------------------------------------------------

class TestFromJSONL:
    def test_basic(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"text": "line one"}\n{"text": "line two"}\n')
        doc = Document.from_jsonl(str(p), text_key="text")
        assert len(doc) == 2
        assert doc[1].text == "line two"

    def test_empty_lines_skipped(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"text": "hello"}\n\n{"text": "world"}\n')
        doc = Document.from_jsonl(str(p), text_key="text")
        assert len(doc) == 2

    def test_empty_text_skipped(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"text": "hello"}\n{"text": ""}\n')
        doc = Document.from_jsonl(str(p), text_key="text")
        assert len(doc) == 1

    def test_label_key(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"text": "verse", "ref": "1:1"}\n')
        doc = Document.from_jsonl(str(p), text_key="text", label_key="ref")
        assert doc[0].label == "1:1"

    def test_title_override(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"text": "verse"}\n')
        doc = Document.from_jsonl(str(p), text_key="text", title="Custom")
        assert doc.title == "Custom"


# ---------------------------------------------------------------------------
# _parse_markdown
# ---------------------------------------------------------------------------

class TestParseMarkdown:
    def test_single_heading(self):
        lines = ["# Book", "verse one", "verse two"]
        passages = _parse_markdown(lines, "line")
        assert len(passages) == 2
        assert all("Book" in (p.label or "") for p in passages)

    def test_heading_hierarchy(self):
        lines = ["# Book", "## Chapter 1", "verse one", "## Chapter 2", "verse two"]
        passages = _parse_markdown(lines, "line")
        assert len(passages) == 2
        assert "Chapter 1" in passages[0].label
        assert "Chapter 2" in passages[1].label

    def test_nested_headings(self):
        lines = ["# H1", "## H2", "### H3", "content"]
        passages = _parse_markdown(lines, "line")
        assert len(passages) == 1
        assert passages[0].label == "H1 > H2 > H3"

    def test_no_headings(self):
        lines = ["line one", "line two"]
        passages = _parse_markdown(lines, "line")
        assert len(passages) == 2
        assert passages[0].label is None

    def test_same_level_heading_replaces(self):
        lines = ["# A", "## Sub1", "text1", "## Sub2", "text2"]
        passages = _parse_markdown(lines, "line")
        assert passages[0].label == "A > Sub1"
        assert passages[1].label == "A > Sub2"

    def test_empty_lines_skipped(self):
        lines = ["# H", "", "   ", "content"]
        passages = _parse_markdown(lines, "line")
        assert len(passages) == 1

    def test_section_splitter_collapses(self):
        lines = ["# Chapter", "line one", "line two"]
        passages = _parse_markdown(lines, "section")
        assert len(passages) == 1
        assert "line one" in passages[0].text

    def test_paragraph_splitter(self):
        lines = ["# H", "line a", "line b", "", "line c"]
        passages = _parse_markdown(lines, "paragraph")
        assert len(passages) == 2


# ---------------------------------------------------------------------------
# _apply_splitter
# ---------------------------------------------------------------------------

class TestApplySplitter:
    def test_line(self):
        result = _apply_splitter(["a", "b", "c"], "line")
        assert result == ["a", "b", "c"]

    def test_paragraph_groups(self):
        result = _apply_splitter(["a", "b", "", "c", "d"], "paragraph")
        assert len(result) == 2
        assert "a" in result[0] and "b" in result[0]
        assert "c" in result[1] and "d" in result[1]

    def test_paragraph_trailing_no_empty(self):
        result = _apply_splitter(["a", "b"], "paragraph")
        assert len(result) == 1

    def test_section_joins_all(self):
        result = _apply_splitter(["word1", "word2", "word3"], "section")
        assert len(result) == 1
        assert "word1" in result[0] and "word3" in result[0]

    def test_callable_splitter(self):
        fn = lambda text: text.split("|")
        result = _apply_splitter(["a|b|c"], fn)
        assert result == ["a", "b", "c"]

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown splitter"):
            _apply_splitter(["text"], "bogus")

    def test_line_strips_whitespace(self):
        result = _apply_splitter(["  hello  "], "line")
        assert result == ["hello"]


# ---------------------------------------------------------------------------
# _path_to_title
# ---------------------------------------------------------------------------

class TestPathToTitle:
    def test_underscores_to_spaces(self):
        assert _path_to_title("my_file.txt") == "my file"

    def test_long_underscore_name(self):
        assert _path_to_title("the_book_of_mormon.md") == "the book of mormon"

    def test_numeric_prefix_stripped(self):
        assert _path_to_title("01_Section_One.txt") == "Section One"

    def test_no_numeric_prefix_keeps_name(self):
        assert _path_to_title("Section_One.txt") == "Section One"

    def test_directory_name(self):
        assert _path_to_title("my_directory") == "my directory"

    def test_full_path_uses_basename(self):
        assert _path_to_title("/path/to/my_file.txt") == "my file"
