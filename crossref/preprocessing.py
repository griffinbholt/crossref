import string
from compose import compose
from typing import Callable


def generate_preprocessing_pipeline(config: dict) -> Callable[[str], str]:
    pipeline: list[Callable[[str], str]] = []
    for preprocessing_module in config['preprocessing']:
        fn_name: str = list(preprocessing_module.keys())[0]
        if fn_name == 'replace_punctuation':
            char_replace: list[tuple[str, str]] = preprocessing_module[fn_name]
            pipeline.append(lambda text: replace_punctuation(text, char_replace))
        elif fn_name == 'remove_all_punctuation_except':
            keep_punctuation: str = preprocessing_module[fn_name]
            pipeline.append(lambda text: remove_punctuation(text, keep_punctuation))
        elif fn_name == 'remove_extra_spaces':
            include: bool = preprocessing_module[fn_name]
            if include:
                pipeline.append(remove_extra_spaces)
        elif fn_name == 'normalize_text':
            include: bool = preprocessing_module[fn_name]
            if include:
                pipeline.append(normalize_text)
        else:
            raise ValueError(f"Unsupported preprocessing function: {fn_name}")
    return compose(*reversed(pipeline))


def replace_punctuation(text: str, char_replace: list[tuple[str, str]]) -> str:
    for char, replace in char_replace:
        text = text.replace(char, replace)
    return text


def remove_punctuation(text: str, keep_punctuation: str = '') -> str:
    remove_set = set(string.punctuation) - set(keep_punctuation)
    return ''.join(' ' if char in remove_set else char for char in text)


def remove_extra_spaces(text: str) -> str:
    return ' '.join(text.split())


def normalize_text(text: str) -> str:
    return text.lower()

# TODO: Remove diacritics - If it removes accents or other diacritical marks.

# TODO: Stopword Remover – If it removes common stopwords.

# TODO: Lemmatizer or Stemmer – If it reduces words to their base or root form.