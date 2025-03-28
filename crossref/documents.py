import os

from typing import Callable, TypedDict
from enum import Enum


class InputFormat(Enum):
    MARKDOWN = 'markdown'
    DIRECTORY = 'directory'

class DocumentJSON(TypedDict, total=False):
    title: str
    content_type: str
    contents: 'Contents'

Contents = list[str | DocumentJSON]


class Document():
    def __init__(
            self,
            pathname: str,
            format: InputFormat,
            preprocess_fn: Callable[[str], str] = lambda x: x
        ):
        parse_methods: dict[InputFormat, Callable[[str], None]] = {
            InputFormat.MARKDOWN: self._parse_markdown_file,
            InputFormat.DIRECTORY: self._parse_directory
        }
        if format not in parse_methods:
            raise ValueError(f"Unsupported input format: {format}")

        self._preprocess_fn = preprocess_fn
        self.structured: DocumentJSON = DocumentJSON()
        self.passages: list[str] = []
        load_document: Callable[[str], None] = parse_methods.get(format)
        load_document(pathname)

    @property
    def title(self) -> str:
        return self.structured.title

    def __len__(self) -> int:
        return len(self.passages)

    def _parse_markdown_file(self, filename: str):
        lines: list[str] = self._read_file(filename)
        self._parse_markdown_section(lines, self.structured)

    def _parse_markdown_section(self, lines: list[str], document: DocumentJSON, i: int = 0, depth: int = 1) -> int:
        document['title'] = lines[i][depth:]
        document['contents'] = []
        i += 1

        while i < len(lines):
            line: str = lines[i]
            if line.startswith('#' * (depth + 1) + ' '):  # One level down
                new_document = DocumentJSON()
                document['contents'].append(new_document)
                i = self._parse_markdown_section(lines, new_document, i, depth + 1)
            elif line.startswith(('#' * depth + ' ', '#' * (depth - 1) + ' ')):  # Same level or one level up
                return i
            else:  # Line of text
                document['contents'].append(line)
                self.passages.append(line)
                i += 1

        return i

    def _parse_directory(self, dirname: str):
        self._parse_subdirectory(dirname, self.structured)

    def _parse_subdirectory(self, dirname: str, document: DocumentJSON):
        document['title'] = self._parse_path_title(dirname)
        document['contents'] = []

        for filename in sorted(os.listdir(dirname)):
            path = os.path.join(dirname, filename)
            if os.path.isdir(path):
                new_document = DocumentJSON()
                document['contents'].append(new_document)
                self._parse_subdirectory(path, new_document)
            elif os.path.isfile(path):
                title: str = self._parse_path_title(path)
                contents = self._read_file(path)
                new_document = DocumentJSON(title=title, contents=contents)
                document['contents'].append(new_document)
                self.passages += contents

    def _parse_path_title(self, path: str) -> str:
        title: str = path.split('/')[-1]
        if title[0].isdigit():
            title = title.split('_', 1)[1]
        if title.endswith('.txt'):
            title = title[:-4]
        return title.replace('_', ' ')

    def _read_file(self, filename: str) -> list[str]:
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
            lines = [self._preprocess_fn(line) for line in lines if line]
        return lines

# def extract_unique_characters(strings):
#     unique_characters = set()
#     for string in strings:
#         unique_characters.update(string)
#     return sorted(unique_characters)

# def main():
#     filename = os.path.expanduser('~/crossref/tests/documents/bookofmormon.md')
#     markdown_doc = Document(filename, InputFormat.MARKDOWN)
#     chars = extract_unique_characters(markdown_doc.passages)
#     print(chars)
    # print_json_to_depth(markdown_doc.structured, depth=4)
    # for passage in markdown_doc.passages[-10:]:
    #     print(f"{passage}\n\n")
    # print(len(markdown_doc))

#     # dirname = os.path.expanduser('~/crossref/documents/Book_of_Mormon')
#     # dir_doc = Document(dirname, InputFormat.DIRECTORY)
#     # print_json_to_depth(dir_doc.structured, depth=4)
#     # for passage in dir_doc.passages[-10:]:
#     #     print(f"{passage}\n\n")
#     # print(len(dir_doc))


# if __name__ == main():
#     main()