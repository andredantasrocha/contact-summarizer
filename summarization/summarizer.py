from summarization.gensim_strategy import GensimStrategy
from summarization.nltk_strategy import NltkStrategy
from summarization.spacy_strategy import SpacyStrategy
from summarization.summa_strategy import SummaStrategy
from summarization.sumy_strategy import SumyStrategy

strategies = [GensimStrategy(), SpacyStrategy(), SummaStrategy(), NltkStrategy(), SumyStrategy()]


class Summarizer:
    def execute(self, text):
        result = {}
        for s in strategies:
            summary = s.summarize_from_text(text)
            result.update(summary)
        return result
