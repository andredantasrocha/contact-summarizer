from summa.summarizer import summarize


class SummaStrategy:
    def __init__(self, ratio=0.2):
        self.__ratio = ratio

    def summarize_from_text(self, text):
        return summarize(text, self.__ratio).strip()
