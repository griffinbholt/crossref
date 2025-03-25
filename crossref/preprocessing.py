import string
from compose import compose
from typing import Callable

def generate_preprocessing_pipeline(config: dict) -> Callable[[str], str]:
    preprocessing_config = config['preprocessing']
    for preprocessing_module in preprocessing_config:
        if preprocessing_module == 'remove_punctuation':
            pass  # TODO


def replace_punctuation(text: str, char_replace: list[tuple[str, str]]) -> str:
    for char, replace in char_replace:
        text = text.replace(char, replace)
    return text

def remove_punctuation(text: str, keep_punctuation: str = '') -> str:
    remove_set = set(string.punctuation) - set(keep_punctuation)
    cleaned_text = ''.join(' ' if char in remove_set else char for char in text)
    return remove_extra_spaces(cleaned_text)

def remove_extra_spaces(text: str) -> str:
    return ' '.join(text.split())

def normalize_text(text: str) -> str:
    return text.lower()

# TODO: Remove diacritics - If it removes accents or other diacritical marks.

# TODO: Stopword Remover – If it removes common stopwords.

# TODO: Lemmatizer or Stemmer – If it reduces words to their base or root form.