from gensim.models import word2vec

class word2vec_embedder:
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
        # model
        # # 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
        self.model.init_sims(replace=True)
        #
        # model_name = '300features_40minwords_10text'
        # # model_name = '300features_50minwords_20text'
        # model.save(model_name)
        # # model.wv.most_similar("/NNG")
        # # model.wv.vocab
    def get_model(self):
        return self.model
