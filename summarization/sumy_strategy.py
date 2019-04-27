from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer


class SumyStrategy:
    def __init__(self):
        self.__language = "english"
        self.__sentences_count = 5

    def summarize_from_text(self, text):
        parser = PlaintextParser.from_string(text, Tokenizer(self.__language))
        return {
            'Luhn': self.__luhn(parser),
            'Edmundson': self.__edmundson(parser),
            'Lsa': self.__lsa(parser),
            'LexRank': self.__lex_rank(parser),
            'TextRank': self.__text_rank(parser),
            'SumBasic': self.__sum_basic(parser),
            'KL-Sum': self.__kl_sum(parser),
        }

    def __luhn(self, parser):
        summarizer = LuhnSummarizer(Stemmer(self.__language))
        # summarizer.stop_words = ("I", "am", "the", "you", "are", "me", "is", "than", "that", "this",)
        final_sentences = summarizer(parser.document, self.__sentences_count)
        return self.__join_sentences(final_sentences)

    def __edmundson(self, parser):
        summarizer = EdmundsonSummarizer(Stemmer(self.__language))
        summarizer.bonus_words = ("deep", "learning", "neural")
        summarizer.stigma_words = ("another", "and", "some", "next",)
        summarizer.null_words = ("another", "and", "some", "next",)
        final_sentences = summarizer(parser.document, self.__sentences_count)
        return self.__join_sentences(final_sentences)

    def __lsa(self, parser):
        summarizer = LsaSummarizer(Stemmer(self.__language))
        summarizer.stop_words = get_stop_words(self.__language)
        final_sentences = summarizer(parser.document, self.__sentences_count)
        return self.__join_sentences(final_sentences)

    def __lex_rank(self, parser):
        summarizer = LexRankSummarizer()
        final_sentences = summarizer(parser.document, self.__sentences_count)
        return self.__join_sentences(final_sentences)

    def __text_rank(self, parser):
        summarizer = TextRankSummarizer(Stemmer(self.__language))
        final_sentences = summarizer(parser.document, self.__sentences_count)
        return self.__join_sentences(final_sentences)

    def __sum_basic(self, parser):
        summarizer = SumBasicSummarizer(Stemmer(self.__language))
        final_sentences = summarizer(parser.document, self.__sentences_count)
        return self.__join_sentences(final_sentences)

    def __kl_sum(self, parser):
        summarizer = KLSummarizer(Stemmer(self.__language))
        final_sentences = summarizer(parser.document, self.__sentences_count)
        return self.__join_sentences(final_sentences)

    def __join_sentences(self, final_sentences):
        summary = ''
        for s in final_sentences:
            summary = '{} {}'.format(summary, str(s))
        return summary
