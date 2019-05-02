from summarization.edmundson_strategy import EdmundsonStrategy
from summarization.gensim_strategy import GensimStrategy
from summarization.kl_sum_strategy import KlSumStrategy
from summarization.lex_rank_strategy import LexRankStrategy
from summarization.lsa_strategy import LsaStrategy
from summarization.luhn_strategy import LuhnStrategy
from summarization.nltk_strategy import NltkStrategy
from summarization.spacy_strategy import SpacyStrategy
from summarization.sum_basic_strategy import SumBasicStrategy
from summarization.summa_strategy import SummaStrategy

from nltk import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from summarization.text_rank_strategy import TextRankStrategy

strategies = {
    'Edmundson': EdmundsonStrategy(),
    'Gensim': GensimStrategy(),
    'KL Sum': KlSumStrategy(),
    'Lex Rank': LexRankStrategy(),
    'LSA': LsaStrategy(),
    'LUHN': LuhnStrategy(),
    'NLTK': NltkStrategy(),
    'Spacy': SpacyStrategy(),
    'Sum Basic': SumBasicStrategy(),
    'Summa': SummaStrategy(),
    'Text Rank': TextRankStrategy(),
}


class Summarizer:
    def execute(self, text):
        result = {'sentiment': self.__sentiment(text), 'summaries': []}
        for name, strategy in strategies.items():
            summary = strategy.summarize_from_text(text)
            result['summaries'].append({'algorithm': name, 'summary': summary})
        return result

    def __sentiment(self, text):
        total = 0
        positive = 0
        neutral = 0
        negative = 0
        analyzer = SentimentIntensityAnalyzer()

        for sentence in sent_tokenize(text):
            compound = analyzer.polarity_scores(sentence)['compound']
            if compound >= 0.05:
                positive += 1
            elif (compound > -0.05) and (compound < 0.05):
                neutral += 1
            else:
                negative += 1
            total += 1

        positive_percent = positive / total * 1.0
        neutral_percent = neutral / total * 1.0
        negative_percent = 1 - positive_percent - neutral_percent
        return {'positive': positive_percent, 'negative': negative_percent, 'neutral': neutral_percent}
