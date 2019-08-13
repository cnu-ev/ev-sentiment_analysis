from soynlp.vectorizer import sent_to_word_contexts_matrix
from eunjeon import Mecab
from glove import Glove
import numpy as np


class Glove_embedder:

    # def __init__(self,train_sentences):
        # self.make_model(train_sentences)
        # self.model = self.load_model()

    def __init__(self):
        self.model = self.load_model()

    def make_model(self,train_sentences):
        # 모델 학습
        tagger = Mecab()
        tokenizer = lambda sentence:list( filter(lambda x: ('JK' in x[1] or 'JC' in x[1] or 'JX' in x[1] or 'VC' in x[1]) == False, tagger.pos(sentence)) )
        corpus = train_sentences
        co_matrix,self.idx2vocab = sent_to_word_contexts_matrix(
            corpus,
            windows=3,
            min_tf=10,
            tokenizer=tokenizer, # (default) lambda x:x.split(),
            dynamic_weight=True,
            verbose=True
        )
        dictionary = {vocab:idx for idx, vocab in enumerate(self.idx2vocab)}
        self.model = Glove(no_components=100, learning_rate=0.05, max_count=30)
        self.model.fit(co_matrix.tocoo(), epochs=5, no_threads=4, verbose=True)
        # 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
        self.model.add_dictionary(dictionary)
        self.model.save("glove_model")

    def get_glove_vector(self):
        zero_vector = np.zeros(100)
        return np.vstack((self.model.word_vectors,zero_vector))

    def get_glove_dictionary(self):
        return self.model.dictionary

    def get_model(self):
        return self.model

    def load_model(self):
        return Glove.load("glove_model")
