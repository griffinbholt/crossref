"""Tests for crossref.preprocessing."""
import pytest

from crossref.preprocessing import (
    replace_punctuation,
    remove_punctuation,
    remove_extra_spaces,
    normalize_text,
    remove_stopwords,
    remove_diacritics,
    lemmatize,
    stem,
    normalize_unicode,
    expand_archaic_forms,
    expand_contractions,
    remove_short_words,
    remove_numbers,
    split_hyphenated_words,
    remove_markup,
    generate_preprocessing_pipeline,
)


# ---------------------------------------------------------------------------
# replace_punctuation
# ---------------------------------------------------------------------------

class TestReplacePunctuation:
    def test_basic(self):
        assert replace_punctuation("it\u2019s fine", [("\u2019", "'")]) == "it's fine"

    def test_multiple_replacements(self):
        result = replace_punctuation("a\u2014b", [("\u2014", " - "), ("\u2019", "'")])
        assert result == "a - b"

    def test_no_match_unchanged(self):
        assert replace_punctuation("hello", [("x", "y")]) == "hello"

    def test_empty_replacement(self):
        assert replace_punctuation("a.b", [(".", "")]) == "ab"


# ---------------------------------------------------------------------------
# remove_punctuation
# ---------------------------------------------------------------------------

class TestRemovePunctuation:
    def test_removes_all_by_default(self):
        result = remove_punctuation("hello, world!")
        assert "," not in result
        assert "!" not in result
        assert "hello" in result

    def test_keep_punctuation(self):
        result = remove_punctuation("it's a test.", keep_punctuation="'")
        assert "'" in result
        assert "." not in result

    def test_replaced_with_space(self):
        # Punctuation becomes a space, not nothing
        result = remove_punctuation("hello,world")
        assert "hello" in result and "world" in result

    def test_no_punctuation_unchanged(self):
        assert remove_punctuation("hello world") == "hello world"


# ---------------------------------------------------------------------------
# remove_extra_spaces
# ---------------------------------------------------------------------------

class TestRemoveExtraSpaces:
    def test_collapses_spaces(self):
        assert remove_extra_spaces("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert remove_extra_spaces("  hello  ") == "hello"

    def test_tabs_and_newlines(self):
        assert remove_extra_spaces("hello\t\nworld") == "hello world"

    def test_already_clean(self):
        assert remove_extra_spaces("hello world") == "hello world"


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_already_lower(self):
        assert normalize_text("hello") == "hello"

    def test_mixed(self):
        assert normalize_text("THE LORD") == "the lord"


# ---------------------------------------------------------------------------
# remove_stopwords
# ---------------------------------------------------------------------------

class TestRemoveStopwords:
    def test_removes_english_stopwords(self):
        result = remove_stopwords("this is a test of the system")
        for word in ("this", "is", "a", "of", "the"):
            assert word not in result.split()

    def test_keeps_content_words(self):
        result = remove_stopwords("the quick fox jumps")
        assert "quick" in result
        assert "fox" in result
        assert "jumps" in result

    def test_custom_stopphrases(self):
        result = remove_stopwords("verily I say unto you", custom_stopphrases={"verily", "unto"})
        assert "verily" not in result
        assert "unto" not in result

    def test_empty_string(self):
        result = remove_stopwords("")
        assert result == ""


# ---------------------------------------------------------------------------
# remove_diacritics
# ---------------------------------------------------------------------------

class TestRemoveDiacritics:
    def test_accented_chars(self):
        assert remove_diacritics("é") == "e"
        assert remove_diacritics("ñ") == "n"
        assert remove_diacritics("ü") == "u"

    def test_plain_ascii_unchanged(self):
        assert remove_diacritics("hello world") == "hello world"

    def test_full_word(self):
        assert remove_diacritics("résumé") == "resume"

    def test_mixed(self):
        result = remove_diacritics("café naïve")
        assert result == "cafe naive"


# ---------------------------------------------------------------------------
# lemmatize
# ---------------------------------------------------------------------------

class TestLemmatize:
    def test_plural_noun(self):
        result = lemmatize("wolves are running")
        assert "wolf" in result

    def test_unchanged_basic(self):
        result = lemmatize("dog cat")
        assert "dog" in result and "cat" in result

    def test_empty(self):
        assert lemmatize("") == ""


# ---------------------------------------------------------------------------
# stem
# ---------------------------------------------------------------------------

class TestStem:
    def test_reduces_words(self):
        result = stem("running jumps happily")
        # All words should be reduced to roots; just check they changed or stayed
        assert len(result.split()) == 3

    def test_running_to_run(self):
        result = stem("running")
        assert result == "run"

    def test_empty(self):
        assert stem("") == ""


# ---------------------------------------------------------------------------
# normalize_unicode
# ---------------------------------------------------------------------------

class TestNormalizeUnicode:
    def test_ligature(self):
        # ﬁ → fi
        result = normalize_unicode("\ufb01le")
        assert result == "file"

    def test_fullwidth(self):
        # Ａ → A
        result = normalize_unicode("\uff21")
        assert result == "A"

    def test_plain_ascii_unchanged(self):
        assert normalize_unicode("hello world") == "hello world"

    def test_fraction(self):
        # ½ (U+00BD) normalizes; the digits 1 and 2 should be present in the result
        result = normalize_unicode("\u00bd")
        assert "1" in result and "2" in result


# ---------------------------------------------------------------------------
# expand_archaic_forms
# ---------------------------------------------------------------------------

class TestExpandArchaicForms:
    def test_hath(self):
        assert expand_archaic_forms("he hath spoken") == "he has spoken"

    def test_thou_thee(self):
        result = expand_archaic_forms("thou art with thee")
        assert "thou" not in result
        assert "thee" not in result

    def test_case_preserved(self):
        # Title-case input → title-case output
        result = expand_archaic_forms("Hath the Lord spoken")
        assert result.startswith("Has")

    def test_no_archaic_unchanged(self):
        assert expand_archaic_forms("hello world") == "hello world"

    def test_verily(self):
        assert "truly" in expand_archaic_forms("verily I say")

    def test_whole_word_only(self):
        # "saith" in "forsaith" should not match
        result = expand_archaic_forms("forsaith")
        assert result == "forsaith"


# ---------------------------------------------------------------------------
# expand_contractions
# ---------------------------------------------------------------------------

class TestExpandContractions:
    def test_cant(self):
        assert expand_contractions("I can't go") == "I cannot go"

    def test_wont(self):
        assert expand_contractions("won't do") == "will not do"

    def test_its(self):
        assert expand_contractions("it's fine") == "it is fine"

    def test_theyre(self):
        assert expand_contractions("they're here") == "they are here"

    def test_no_contraction(self):
        assert expand_contractions("hello world") == "hello world"

    def test_longer_before_shorter(self):
        # "shouldn't" should expand to "should not", not "should n't"
        result = expand_contractions("shouldn't")
        assert result == "should not"


# ---------------------------------------------------------------------------
# remove_short_words
# ---------------------------------------------------------------------------

class TestRemoveShortWords:
    def test_removes_single_chars(self):
        result = remove_short_words("I a am going")
        assert "I" not in result.split()
        assert "a" not in result.split()

    def test_default_min_length_2(self):
        result = remove_short_words("go to the store")
        # "to" (2 chars) should be kept by default
        assert "to" in result.split()

    def test_custom_min_length(self):
        result = remove_short_words("and or the lord", min_length=4)
        assert "and" not in result.split()
        assert "lord" in result.split()

    def test_empty(self):
        assert remove_short_words("") == ""


# ---------------------------------------------------------------------------
# remove_numbers
# ---------------------------------------------------------------------------

class TestRemoveNumbers:
    def test_removes_standalone_digits(self):
        result = remove_numbers("verse 42 and chapter 1")
        assert "42" not in result
        assert "1" not in result

    def test_embedded_digits_kept(self):
        result = remove_numbers("covid19 and mp3")
        assert "covid19" in result
        assert "mp3" in result

    def test_no_numbers_unchanged(self):
        result = remove_numbers("hello world")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# split_hyphenated_words
# ---------------------------------------------------------------------------

class TestSplitHyphenatedWords:
    def test_basic(self):
        assert split_hyphenated_words("well-known") == "well known"

    def test_multiple(self):
        result = split_hyphenated_words("well-known self-taught")
        assert result == "well known self taught"

    def test_leading_hyphen_unchanged(self):
        result = split_hyphenated_words("-foo")
        assert result == "-foo"

    def test_no_hyphen_unchanged(self):
        assert split_hyphenated_words("hello world") == "hello world"


# ---------------------------------------------------------------------------
# remove_markup
# ---------------------------------------------------------------------------

class TestRemoveMarkup:
    def test_markdown_link(self):
        result = remove_markup("[click here](http://example.com)")
        assert result == "click here"
        assert "http" not in result

    def test_markdown_image_removed(self):
        result = remove_markup("![alt text](image.png) rest")
        assert "alt text" not in result
        assert "rest" in result

    def test_bold(self):
        assert remove_markup("**bold**") == "bold"
        assert remove_markup("__bold__") == "bold"

    def test_italic(self):
        assert remove_markup("*italic*") == "italic"
        assert remove_markup("_italic_") == "italic"

    def test_inline_code(self):
        assert remove_markup("`code`") == "code"

    def test_html_tag(self):
        result = remove_markup("<p>paragraph</p>")
        assert "<p>" not in result
        assert "paragraph" in result

    def test_plain_text_unchanged(self):
        assert remove_markup("hello world") == "hello world"


# ---------------------------------------------------------------------------
# generate_preprocessing_pipeline
# ---------------------------------------------------------------------------

class TestGeneratePreprocessingPipeline:
    def _pipeline(self, steps):
        return generate_preprocessing_pipeline({"preprocessing": steps})

    def test_empty_pipeline_is_identity(self):
        fn = generate_preprocessing_pipeline({})
        assert fn("Hello World") == "Hello World"

    def test_boolean_flag_normalize_text(self):
        fn = self._pipeline([{"normalize_text": True}])
        assert fn("HELLO WORLD") == "hello world"

    def test_boolean_flag_false_skips(self):
        fn = self._pipeline([{"normalize_text": False}])
        assert fn("HELLO WORLD") == "HELLO WORLD"

    def test_boolean_flag_remove_extra_spaces(self):
        fn = self._pipeline([{"remove_extra_spaces": True}])
        assert fn("hello   world") == "hello world"

    def test_parameterized_replace_punctuation(self):
        fn = self._pipeline([{"replace_punctuation": [["\u2019", "'"]]}])
        assert fn("it\u2019s") == "it's"

    def test_parameterized_remove_all_punctuation_except(self):
        fn = self._pipeline([{"remove_all_punctuation_except": "'"}])
        result = fn("hello, world!")
        assert "," not in result
        assert "!" not in result
        assert "'" in fn("it's")

    def test_parameterized_remove_stopwords_none(self):
        fn = self._pipeline([{"remove_stopwords": None}])
        result = fn("the quick brown fox")
        assert "the" not in result.split()

    def test_parameterized_remove_short_words(self):
        fn = self._pipeline([{"remove_short_words": 3}])
        result = fn("I am going")
        assert "I" not in result.split()
        assert "am" not in result.split()
        assert "going" in result

    def test_multiple_steps_applied_in_order(self):
        fn = self._pipeline([
            {"normalize_text": True},
            {"remove_extra_spaces": True},
        ])
        assert fn("  HELLO   WORLD  ") == "hello world"

    def test_unknown_function_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            self._pipeline([{"totally_fake_function": True}])

    def test_expand_archaic_forms(self):
        fn = self._pipeline([{"expand_archaic_forms": True}])
        assert "has" in fn("he hath spoken")

    def test_expand_contractions(self):
        fn = self._pipeline([{"expand_contractions": True}])
        assert fn("can't") == "cannot"

    def test_normalize_unicode(self):
        fn = self._pipeline([{"normalize_unicode": True}])
        assert fn("\ufb01le") == "file"

    def test_remove_numbers(self):
        fn = self._pipeline([{"remove_numbers": True}])
        assert "42" not in fn("verse 42")

    def test_remove_markup(self):
        fn = self._pipeline([{"remove_markup": True}])
        assert fn("**bold**") == "bold"

    def test_split_hyphenated_words(self):
        fn = self._pipeline([{"split_hyphenated_words": True}])
        assert fn("well-known") == "well known"
