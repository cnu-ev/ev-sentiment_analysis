from soynlp.vectorizer import sent_to_word_contexts_matrix
from eunjeon import Mecab
from glove import Glove
import numpy as np


def makeFeatureVec(words, model ,num_features):
    # """
    # 주어진 문장에서 단어 벡터의 평균을 구하는 함수
    # """
    # 속도를 위해 0으로 채운 배열로 초기화 한다.
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    # Index2word는 모델의 사전에 있는 단어명을 담은 리스트이다.
    # 속도를 위해 set 형태로 초기화 한다.
    index2word_set = set(model.wv.index2word)
    # 루프를 돌며 모델 사전에 포함이 되는 단어라면 피처에 추가한다.
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 결과를 단어수로 나누어 평균을 구한다.
    if(nwords):
        featureVec = np.divide(featureVec,nwords)
    else:
        featureVec = np.zeros((num_features,),dtype="float32")
    return featureVec

def getAvgFeatureVecs(reviews,model,num_features):
    # 리뷰 단어 목록의 각각에 대한 평균 feature 벡터를 계산하고
    # 2D numpy 배열을 반환한다.
    # 카운터를 초기화 한다.
    counter = 0.
    # 속도를 위해 2D 넘파이 배열을 미리 할당한다.
    reviewFeatureVecs = np.zeros(
        (len(reviews),num_features),dtype="float32")

    for review in reviews:
          # 매 1000개 리뷰마다 상태를 출력
          if counter%1000. == 0.:
              print("Review %d of %d" % (counter, len(reviews)))
          # 평균 피처 벡터를 만들기 위해 위에서 정의한 함수를 호출한다.
          reviewFeatureVecs[int(counter)] = makeFeatureVec(review,model,num_features)
          # 카운터를 증가시킨다.
          counter = counter + 1.
    return reviewFeatureVecs


class Word2vec_embedder:

    def __init__(self,train_sentences):
        self.make_model(train_sentences)
        # self.model = self.load_model()

    # def __init__(self):
        # self.model = self.load_model('word2vec.model')

    def make_model(self,train_sentences):
        # 모델 학습
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


    def get_model(self):
        return self.model

    def load_model(self):
        return Glove.load("glove_model")
