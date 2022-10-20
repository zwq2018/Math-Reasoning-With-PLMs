import re
class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}       #
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if word == 'PI':
                word = '3.14'
            if word =='1':
                word = '1.0'

            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.index2word.append(word)
                self.n_words += 1





    def build_output_lang_for_tree(self):  # build the output lang vocab and dict

        self.num_start = len(['+', '-', '-_rev', '*', '/', '/_rev'])      #

        self.index2word =  ['+', '-', '-_rev', '*', '/', '/_rev']  + ['1.0','3.14'] + ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r'] + ['(',')']+["UNK"]
        self.n_words = len(self.index2word)    # 

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

