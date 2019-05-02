from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.utils import get_stop_words


class LuhnStrategy:
    def __init__(self, language='english', sentences_count=5):
        self.__language = language
        self.__sentences_count = sentences_count

    def summarize_from_text(self, text):
        parser = PlaintextParser.from_string(text, Tokenizer(self.__language))
        return self.__summarize(parser),

    def __summarize(self, parser):
        summarizer = LuhnSummarizer(Stemmer(self.__language))
        summarizer.stop_words = get_stop_words(self.__language)
        final_sentences = summarizer(parser.document, self.__sentences_count)
        return self.__join_sentences(final_sentences)

    def __join_sentences(self, final_sentences):
        summary = ''
        for s in final_sentences:
            summary = '{} {}'.format(summary, str(s))
        return summary.strip()
