from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize


class NltkStrategy:
    def summarize_from_text(self, text):
        # 1 Create the word frequency table
        freq_table = self.__create_frequency_table(text)

        '''
        We already have a sentence tokenizer, so we just need 
        to run the sent_tokenize() method to create the array of sentences.
        '''

        # 2 Tokenize the sentences
        sentences = sent_tokenize(text)

        # 3 Important Algorithm: score the sentences
        sentence_scores = self.__score_sentences(sentences, freq_table)

        # 4 Find the threshold
        threshold = self.__find_average_score(sentence_scores)

        # 5 Important Algorithm: Generate the summary
        summary = self.__generate_summary(sentences, sentence_scores, 1.3 * threshold)

        return {'NLTK': summary}

    def __create_frequency_table(self, text) -> dict:
        """
        we create a dictionary for the word frequency table.
        For this, we should only use the words that are not part of the stopWords array.

        Removing stop words and making frequency table
        Stemmer - an algorithm to bring words to its root word.
        :rtype: dict
        """
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text)
        ps = PorterStemmer()

        freq_table = dict()
        for word in words:
            word = ps.stem(word)
            if word in stop_words:
                continue
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        return freq_table

    def __score_sentences(self, sentences, freq_table) -> dict:
        """
        score a sentence by its words
        Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
        :rtype: dict
        """

        sentence_value = dict()

        for sentence in sentences:
            word_count_in_sentence = (len(word_tokenize(sentence)))
            word_count_in_sentence_except_stop_words = 0
            for wordValue in freq_table:
                if wordValue in sentence.lower():
                    word_count_in_sentence_except_stop_words += 1
                    if sentence[:10] in sentence_value:
                        sentence_value[sentence[:10]] += freq_table[wordValue]
                    else:
                        sentence_value[sentence[:10]] = freq_table[wordValue]

            sentence_value[sentence[:10]] = sentence_value[sentence[:10]] / word_count_in_sentence_except_stop_words

            '''
            Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences. 
            To solve this, we're dividing every sentence score by the number of words in the sentence.
            
            Note that here sentence[:10] is the first 10 character of any sentence, this is to save memory while saving keys of
            the dictionary.
            '''

        return sentence_value

    def __find_average_score(self, sentence_value) -> int:
        """
        Find the average score from the sentence value dictionary
        :rtype: int
        """
        sum_values = 0
        for entry in sentence_value:
            sum_values += sentence_value[entry]

        # Average value of a sentence from original text
        average = (sum_values / len(sentence_value))

        return average

    def __generate_summary(self, sentences, sentence_value, threshold):
        sentence_count = 0
        summary = ''

        for sentence in sentences:
            if sentence[:10] in sentence_value and sentence_value[sentence[:10]] >= threshold:
                summary += " " + sentence
                sentence_count += 1

        return summary
