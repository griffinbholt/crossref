import string

from compose import compose
from nltk.corpus import stopwords
from typing import Callable


def generate_preprocessing_pipeline(config: dict) -> Callable[[str], str]:
    pipeline: list[Callable[[str], str]] = []
    for preprocessing_module in (config.get('preprocessing') or []):
        fn_name: str = list(preprocessing_module.keys())[0]
        if fn_name == 'replace_punctuation':
            char_replace: list[tuple[str, str]] = preprocessing_module[fn_name]
            pipeline.append(lambda text, cr=char_replace: replace_punctuation(text, cr))
        elif fn_name == 'remove_all_punctuation_except':
            keep_punctuation: str = preprocessing_module[fn_name]
            pipeline.append(lambda text, kp=keep_punctuation: remove_punctuation(text, kp))
        elif fn_name == 'remove_extra_spaces':
            include: bool = preprocessing_module[fn_name]
            if include:
                pipeline.append(remove_extra_spaces)
        elif fn_name == 'normalize_text':
            include: bool = preprocessing_module[fn_name]
            if include:
                pipeline.append(normalize_text)
        elif fn_name == 'remove_stopwords':
            custom_stopphrases: set[str] = set(preprocessing_module[fn_name] or [])
            pipeline.append(lambda text, cs=custom_stopphrases: remove_stopwords(text, cs))
        else:
            raise ValueError(f"Unsupported preprocessing function: {fn_name}")
    if not pipeline:
        return lambda text: text
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


def remove_stopwords(text: str, custom_stopphrases: set[str] | None = None) -> str:
    stopphrases = set(stopwords.words('english'))
    if custom_stopphrases:
        stopphrases |= custom_stopphrases
    # Pad with spaces so phrases at the start/end of the string are also matched.
    text = f' {text} '
    for phrase in sorted(stopphrases, key=len, reverse=True):
        text = text.replace(f' {phrase} ', ' ')
    return ' '.join(word for word in text.split() if word not in stopphrases)


# TODO: Remove diacritics - If it removes accents or other diacritical marks.

# TODO: Lemmatizer or Stemmer – If it reduces words to their base or root form.
