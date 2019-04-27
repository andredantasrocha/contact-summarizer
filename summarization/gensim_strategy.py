from gensim.summarization.summarizer import summarize


class GensimStrategy:
    def __init__(self, ratio=0.2):
        self.__ratio = ratio

    def summarize_from_text(self, text):
        return {'Gensim': (summarize(text, self.__ratio))}
