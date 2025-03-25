import sys

def main(title: str, filename: str):
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        inlines: list[str] = [line.strip() for line in file.readlines()]
    
    outlines: list[str] = [f"# {title}\n\n"]
    current_book: str = ''
    current_chapter: int = 0
    for inline in inlines:
        if not inline:
            continue
        book, chapter, _, text = split_line(inline)
        if book != current_book:
            current_book = book
            current_chapter = 0
            outlines.append(f"## {book}\n")
            outlines.append('\n')
        if chapter != current_chapter:
            current_chapter = chapter
            outlines.append(f"### Chapter {chapter}\n\n")
        outlines.append(f"{text}\n\n")

    with open(filename.replace('.txt', '.md'), mode='w', encoding='utf-8') as file:
        file.writelines(outlines)


def split_line(line: str) -> tuple[str, int, int, str]:
    label, text = line.split('\t')
    components = label.split(' ')
    book = ' '.join(components[:-1])
    chapter, verse = components[-1].split(':')
    return book, int(chapter), int(verse), text

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])