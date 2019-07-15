from gensim.models import word2vec
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
    # 파라메터값 지정
    num_features = 300 # 문자 벡터 차원 수
    min_word_count = 40 # 최소 문자 수
    num_workers = 4 # 병렬 처리 스레드 수
    context = 10 # 문자열 창 크기
    downsampling = 1e-3 # 문자 빈도 수 Downsample
    def __init__(self,train_sentences):
        self.make_model(train_sentences)


    def make_model(self,train_sentences):
        # 모델 학습
        self.model = word2vec.Word2Vec(train_sentences,
                                  workers=self.num_workers,
                                  size=self.num_features,
                                  min_count=self.min_word_count,
                                  window=self.context,
                                  sample=self.downsampling)
        # 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
        self.model.init_sims(replace=True)

    def get_model(self):
        return self.model
