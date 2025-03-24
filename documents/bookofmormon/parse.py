#!/usr/bin/env python3

import json
import os

BOM_DIR = os.path.expanduser("~/crossref/documents/bookofmormon")

BOOKS = [
    'Title Page',
    '1 Nephi',
    '2 Nephi',
    'Jacob',
    'Enos',
    'Jarom',
    'Omni',
    'Words of Mormon',
    'Mosiah',
    'Alma',
    'Helaman',
    '3 Nephi',
    '4 Nephi',
    'Mormon',
    'Ether',
    'Moroni'
]

def main():
    book_of_mormon = {
        'title': 'Book of Mormon',
        'contents': [ {'title': book, 'contents': [] } for book in BOOKS]
    }

    for i, book in enumerate(BOOKS):
        # Read in the verses
        infilename: str = "{:02d}_{}.txt".format(i, book.replace(' ', '_'))
        with open(f"{BOM_DIR}/{infilename}", 'r') as file:
            verses: list[str] = file.readlines()
        
        # Create the directory
        outdir: str = "{:02d}_{}".format(i, book.replace(' ', '_'))
        os.makedirs(f"{BOM_DIR}/{outdir}", exist_ok=True)

        book_of_mormon['contents'][i]['contents']

        prev_chapter: int = 0
        for verse in verses:
            num, text = verse.split(' ', 1)
            curr_chapter, curr_verse = num.split(':')
            curr_chapter, curr_verse = int(curr_chapter), int(curr_verse)
            if curr_chapter != prev_chapter:
                book_of_mormon['contents'][i]['contents'].append(
                    {
                        'title': f"Chapter {curr_chapter}",
                        'contents': []
                    }
                )
                prev_chapter = curr_chapter
            book_of_mormon['contents'][i]['contents'][curr_chapter - 1]['contents'].append(text)
    
    with open(f"{BOM_DIR}/book_of_mormon.json", 'w') as file:
        json.dump(book_of_mormon, file, indent=4)


if __name__ == "__main__":
    main()