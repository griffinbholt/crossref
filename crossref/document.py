import os

from typing import Callable, TypedDict
from enum import Enum

from utils import print_json_to_depth

class InputFormat(Enum):
    MARKDOWN = 0
    DIRECTORY = 1

class DocumentJSON(TypedDict, total=False):
    title: str
    content_type: str
    contents: 'Contents'

Contents = list[str | DocumentJSON]


class Document():
    def __init__(self, path: str, input_format: InputFormat):
        self.structured: DocumentJSON = DocumentJSON()
        self.passages: list[str] = []

        parse_methods: dict[InputFormat, Callable[[str], None]] = {
            InputFormat.MARKDOWN: self._parse_markdown_file,
            InputFormat.DIRECTORY: self._parse_directory
        }
        load_document: Callable[[str], None] = parse_methods.get(input_format)
        if load_document is None:
            raise ValueError(f"Unsupported input format: {input_format}")
        load_document(path)

    @property
    def title(self) -> str:
        return self.structured.title

    def __len__(self) -> int:
        return len(self.passages)

    def _parse_markdown_file(self, filename: str):
        with open(filename, 'r') as file:
            lines = file.readlines()
        self._parse_markdown_section(lines, self.structured)

    def _parse_markdown_section(self, lines: list[str], document: DocumentJSON, i: int = 0, depth: int = 1) -> int:
        document['title'] = lines[i][depth:].strip()
        document['contents'] = []
        i += 1

        while i < len(lines):
            stripped: str = lines[i].strip()
            if not stripped:  # Empty line
                i += 1
            elif stripped.startswith('#' * (depth + 1) + ' '):  # One level down
                new_document = DocumentJSON()
                document['contents'].append(new_document)
                i = self._parse_markdown_section(lines, new_document, i, depth + 1)
            elif stripped.startswith(('#' * depth + ' ', '#' * (depth - 1) + ' ')):  # Same level or one level up
                return i
            else:  # Line of text
                document['contents'].append(stripped)
                self.passages.append(stripped)
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
                with open(path, 'r') as file:
                    contents = [line.strip() for line in file.readlines()]
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

def save_to_subdirectory(parentdir: str, document: DocumentJSON, j: int = 0):
    title = document['title'].replace(' ', '_')
    contents = document['contents']

    for i, content in enumerate(contents):
        if isinstance(content, str):
            with open(f"{parentdir}/{j:02d}_{title}.txt", 'a+') as file:
                file.write(f"{content}\n")
        else:
            subdir = f"{parentdir}/{j:02d}_{title}"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            save_to_subdirectory(subdir, content, i)


def main():
    import os
    # filename = os.path.expanduser('~/crossref/documents/bookofmormon.md')
    # doc = Document(filename, InputFormat.MARKDOWN)
    # save_to_subdirectory('/Users/griffinbholt/crossref/documents', doc.structured)

    dirname = os.path.expanduser('~/crossref/documents/Book_of_Mormon')
    doc = Document(dirname, InputFormat.DIRECTORY)
    print_json_to_depth(doc.structured, depth=4)
    # for passage in doc.passages[-10:]:
    #     print(f"{passage}\n\n")
    # print(len(doc.passages))


if __name__ == main():
    main()