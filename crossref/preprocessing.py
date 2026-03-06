import logging
import re
import string
import unicodedata

from compose import compose
from nltk.corpus import stopwords
from typing import Callable

logger = logging.getLogger(__name__)


def replace_punctuation(text: str, char_replace: list[tuple[str, str]]) -> str:
    """Replace specific characters with substitutes.

    Useful for normalizing typographic variants before further processing,
    e.g. replacing curly quotes with straight quotes or em-dashes with spaces.

    Args:
        text: Input text.
        char_replace: List of (old, new) pairs applied in order.
    """
    for char, replace in char_replace:
        text = text.replace(char, replace)
    return text


def remove_punctuation(text: str, keep_punctuation: str = '') -> str:
    """Remove punctuation characters, replacing them with spaces.

    Removes all characters in string.punctuation except those explicitly
    listed in keep_punctuation. Replaced characters become spaces rather
    than being deleted outright, so word boundaries are preserved.

    Args:
        text: Input text.
        keep_punctuation: String of punctuation characters to leave intact.
                          E.g. "'-" to keep hyphens and apostrophes.
    """
    remove_set = set(string.punctuation) - set(keep_punctuation)
    return ''.join(' ' if char in remove_set else char for char in text)


def remove_extra_spaces(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip leading/trailing whitespace."""
    return ' '.join(text.split())


def normalize_text(text: str) -> str:
    """Lowercase all characters."""
    return text.lower()


def remove_stopwords(text: str, custom_stopphrases: set[str] | None = None) -> str:
    """Remove English stopwords and optional custom stop-phrases.

    Uses NLTK's English stopword list as the base. Custom stop-phrases (which
    may be multi-word) are matched as whole token sequences and removed first,
    longest-match first, before single-word stopword removal.

    Args:
        text: Input text.
        custom_stopphrases: Additional words or phrases to remove, e.g.
                            {"saith the lord", "verily"}.
    """
    stopphrases = set(stopwords.words('english'))
    if custom_stopphrases:
        stopphrases |= custom_stopphrases
    # Pad with spaces so phrases at the start/end of the string are also matched.
    text = f' {text} '
    for phrase in sorted(stopphrases, key=len, reverse=True):
        text = text.replace(f' {phrase} ', ' ')
    return ' '.join(word for word in text.split() if word not in stopphrases)


def remove_diacritics(text: str) -> str:
    """Strip accent marks and other diacritical marks from characters.

    Decomposes the text using Unicode NFD normalization (which separates base
    characters from their combining marks), then discards all combining mark
    code points (Unicode category 'Mn'). This maps e.g. é→e, ñ→n, ü→u.

    Has no effect on characters that have no decomposable diacritics.
    """
    normalized = unicodedata.normalize('NFD', text)
    return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')


def lemmatize(text: str) -> str:
    """Reduce each word to its dictionary base form using NLTK's WordNetLemmatizer.

    Lemmatization maps inflected word forms to their canonical lemma:
    e.g. "running"→"run", "better"→"good" (with POS context), "wolves"→"wolf".
    Without POS tagging, words are assumed to be nouns, so verb forms may not
    fully reduce (e.g. "running" stays "running" unless tagged as a verb).

    More linguistically accurate than stemming but slower, and requires the
    NLTK 'wordnet' corpus (downloaded automatically on first use).
    """
    import nltk
    nltk.download('wordnet', quiet=True)
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())


def stem(text: str) -> str:
    """Reduce each word to its stem using the Porter Stemmer.

    Stemming aggressively strips suffixes to a common root form:
    e.g. "running"→"run", "happiness"→"happi", "generously"→"generous".
    The output may not be a real word, but words sharing a root will map
    to the same stem, which can improve recall for syntactic metrics.

    Faster than lemmatization and requires no extra downloads.
    """
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    return ' '.join(stemmer.stem(word) for word in text.split())


def normalize_unicode(text: str) -> str:
    """Apply NFKC Unicode normalization.

    Collapses compatibility variants to their canonical forms beyond what
    remove_diacritics handles: ligatures (ﬁ→fi), fullwidth characters (Ａ→A),
    superscripts (²→2), fractions (½→1/2), and other presentation forms.

    Safe to run before any other step; has no visible effect on plain ASCII.
    """
    return unicodedata.normalize('NFKC', text)


# ---------------------------------------------------------------------------
# Archaic form expansion (KJV English → modern English)
# ---------------------------------------------------------------------------

_ARCHAIC_FORMS: dict[str, str] = {
    # Verb forms
    'hath': 'has', 'doth': 'does', 'doeth': 'does',
    'hast': 'have', 'dost': 'do',
    'wast': 'were', 'wert': 'were',
    'shalt': 'shall', 'wilt': 'will',
    'wouldst': 'would', 'wouldest': 'would',
    'shouldst': 'should', 'couldst': 'could',
    'canst': 'can', 'mayst': 'may', 'mightest': 'might',
    'saith': 'says', 'sayest': 'say',
    'spake': 'spoke',
    'knoweth': 'knows', 'knowest': 'know',
    'seeth': 'sees', 'seest': 'see',
    'cometh': 'comes', 'goeth': 'goes',
    'giveth': 'gives', 'taketh': 'takes',
    'maketh': 'makes', 'speaketh': 'speaks',
    'leadeth': 'leads', 'bringeth': 'brings',
    'keepeth': 'keeps', 'walketh': 'walks',
    # Pronouns
    'thou': 'you', 'thee': 'you',
    'thy': 'your', 'thine': 'your',
    'ye': 'you',
    # Adverbs / conjunctions / prepositions
    'whilst': 'while', 'amongst': 'among',
    'whence': 'where', 'wherefore': 'therefore',
    'thereof': 'of it', 'therein': 'in it',
    'thereon': 'on it', 'thereunto': 'to it',
    'whereby': 'by which', 'wherein': 'in which', 'whereof': 'of which',
    # Affirmations / interjections
    'verily': 'truly', 'yea': 'yes', 'nay': 'no',
    'behold': 'look',
    # Nouns
    'brethren': 'brothers',
}

_ARCHAIC_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in _ARCHAIC_FORMS) + r')\b',
    re.IGNORECASE,
)


def expand_archaic_forms(text: str) -> str:
    """Replace KJV / Early Modern English forms with their modern equivalents.

    Covers archaic verb conjugations (hath→has, doth→does, spake→spoke),
    second-person pronouns (thou→you, thee→you, thy→your, thine→your, ye→you),
    archaic adverbs and conjunctions (verily→truly, whilst→while, wherefore→therefore),
    and common archaic nouns (brethren→brothers).

    Matching is case-insensitive; title-case originals produce title-case output
    (e.g. "Hath"→"Has"). Replacements are whole-word only (word-boundary anchored).

    Useful when cross-referencing KJV scripture against modern translations,
    since syntactic metrics treat "doth" and "does" as completely different tokens.
    """
    def _replace(m: re.Match) -> str:
        word = m.group(0)
        replacement = _ARCHAIC_FORMS[word.lower()]
        return replacement.capitalize() if word[0].isupper() else replacement

    return _ARCHAIC_PATTERN.sub(_replace, text)


# ---------------------------------------------------------------------------
# Contraction expansion
# ---------------------------------------------------------------------------

_CONTRACTIONS: dict[str, str] = {
    "can't": "cannot", "won't": "will not",
    "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not",
    "shouldn't": "should not", "wouldn't": "would not",
    "couldn't": "could not", "mightn't": "might not",
    "mustn't": "must not", "needn't": "need not",
    "i'm": "i am", "i've": "i have",
    "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have",
    "you'll": "you will", "you'd": "you would",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "we've": "we have",
    "we'll": "we will", "we'd": "we would",
    "they're": "they are", "they've": "they have",
    "they'll": "they will", "they'd": "they would",
    "let's": "let us", "that's": "that is",
    "who's": "who is", "what's": "what is",
    "there's": "there is", "here's": "here is",
    "where's": "where is", "how's": "how is",
}

_CONTRACTION_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)


def expand_contractions(text: str) -> str:
    """Expand common English contractions to their full forms.

    E.g. "can't"→"cannot", "it's"→"it is", "they've"→"they have".

    Matching is case-insensitive. Possessive "'s" (e.g. "John's") is not
    expanded since it is not a contraction. Longer contractions are matched
    before shorter ones to avoid partial substitution.
    """
    return _CONTRACTION_PATTERN.sub(lambda m: _CONTRACTIONS[m.group(0).lower()], text)


def remove_short_words(text: str, min_length: int = 2) -> str:
    """Remove words shorter than min_length characters.

    Useful as a complement to stopword removal for dropping single-letter
    tokens (e.g. "a", "I") or other very short words that add noise.

    Args:
        text: Input text.
        min_length: Minimum word length to keep (inclusive). Default 2.
    """
    return ' '.join(w for w in text.split() if len(w) >= min_length)


def remove_numbers(text: str) -> str:
    """Remove standalone digit sequences from the text.

    Strips tokens composed entirely of digits (e.g. verse numbers that bleed
    into passage text after splitting). Digits embedded in words (e.g.
    "mp3", "covid19") are left intact. Whitespace left by removed tokens
    is not collapsed — run remove_extra_spaces afterwards if needed.
    """
    return re.sub(r'\b\d+\b', '', text)


def split_hyphenated_words(text: str) -> str:
    """Replace hyphens between word characters with spaces.

    Splits compound words like "well-known"→"well known" so that component
    words are treated as independent tokens by downstream metrics. Only
    hyphens flanked by word characters are affected; em-dashes and
    leading/trailing hyphens are left unchanged.
    """
    return re.sub(r'(\w)-(\w)', r'\1 \2', text)


def remove_markup(text: str) -> str:
    """Strip common inline markup, preserving the underlying text content.

    Handles:
    - Markdown links ([text](url)) → text
    - Markdown images (![alt](url)) → removed
    - Bold/italic (**text**, *text*, __text__, _text_) → text
    - Inline code (`text`) → text
    - HTML tags (<tag>) → removed
    """
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)           # images: remove
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)       # links: keep text
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)              # bold **
    text = re.sub(r'__([^_]+)__', r'\1', text)                  # bold __
    text = re.sub(r'\*([^*]+)\*', r'\1', text)                  # italic *
    text = re.sub(r'_([^_]+)_', r'\1', text)                    # italic _
    text = re.sub(r'`([^`]+)`', r'\1', text)                    # inline code
    text = re.sub(r'<[^>]+>', '', text)                         # HTML tags
    return text


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

# Functions where the config value is the argument (not a boolean flag).
_PARAMETERIZED_FNS: dict[str, Callable] = {
    'replace_punctuation':           lambda v: lambda text: replace_punctuation(text, v),
    'remove_all_punctuation_except': lambda v: lambda text: remove_punctuation(text, v),
    'remove_stopwords':              lambda v: lambda text: remove_stopwords(text, set(v or [])),
    'remove_short_words':            lambda v: lambda text: remove_short_words(text, v),
}


def generate_preprocessing_pipeline(config: dict) -> Callable[[str], str]:
    pipeline: list[Callable[[str], str]] = []
    step_names: list[str] = []
    for module in (config.get('preprocessing') or []):
        fn_name, value = next(iter(module.items()))
        if fn_name in _PARAMETERIZED_FNS:
            pipeline.append(_PARAMETERIZED_FNS[fn_name](value))
            step_names.append(fn_name)
        elif fn := globals().get(fn_name):
            if value:
                pipeline.append(fn)
                step_names.append(fn_name)
        else:
            raise ValueError(f"Unsupported preprocessing function: {fn_name!r}")
    if not pipeline:
        logger.debug("No preprocessing steps configured — using identity pipeline")
        return lambda text: text
    logger.debug("Preprocessing pipeline (%d steps): %s", len(step_names), " → ".join(step_names))
    return compose(*reversed(pipeline))
