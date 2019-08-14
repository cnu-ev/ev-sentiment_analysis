from data_manager import Data_manager
# from data_preprocessor import Data_preprocessor
from glove_embedder import Glove_embedder
from sentiment_analysis_model import Sentiment_analysis_model
import pickle

class Sentiment_analysis:
    def __init__(self):
        #data_manager 생성
        # self.data_manager = Data_manager()
        #
        # train_sentences = self.data_manager.get_train_sentences()
        # train_tags = self.data_manager.get_train_tags()

        #학습 문장으로 word2_vec_embedder 생성
        self.glove_embedder = Glove_embedder()
        self.glove_dictionary = self.glove_embedder.get_glove_dictionary()
        self.glove_vector = self.glove_embedder.get_glove_vector()
        #indexing_embedding한 벡터

        # train_sentences= [ data_preprocessor.indexing_embedding(sentence,glove_dictionary) for sentence in train_sentences]
        # train_sentences_torch = torch.tensor(train_sentences)
        # train_tags_torch = torch.tensor([int(x) for x in train_tags])
        #embedding 된 vector_set과 정답지로 Sentiment_analysis 모델 생성
        self.sentiment_analysis_model = Sentiment_analysis_model(self.glove_dictionary,self.glove_vector)

    def Sentiment_analysis_predict_pos_neg(self,sentence):
        # sentence, score = self.sentiment_analysis_model.predict_pos_neg_by_loaded_model(sentence,self.word2vec_embedder.model,self.word2vec_embedder.num_features)
        sentence, score = self.sentiment_analysis_model.predict_pos_neg_loading_model(sentence,self.glove_dictionary,self.glove_vector)

        return sentence,score

    def training_model(self):
        #data_manager 생성
        self.data_manager = Data_manager()

        train_sentences = self.data_manager.get_train_sentences()
        train_tags = self.data_manager.get_train_tags()

        #학습 문장으로 word2_vec_embedder 생성
        #glove 임베딩에 넣는 문장들은 tagger 안된 문장 넣어야됨
        self.glove_embedder = Glove_embedder(train_sentences)
        self.glove_dictionary = self.glove_embedder.get_glove_dictionary()
        #indexing_embedding한 벡터

        train_sentences= [ data_preprocessor.indexing_embedding(sentence,glove_dictionary) for sentence in train_sentences]
        train_sentences_torch = torch.tensor(train_sentences)
        train_tags_torch = torch.tensor([int(x) for x in train_tags])
        #embedding 된 vector_set과 정답지로 Sentiment_analysis 모델 생성
        self.sentiment_analysis_model = Sentiment_analysis_model(train_sentences_torch,train_tags_torch)


    def update_model(self):
        #data_manager 생성
        self.data_manager = Data_manager()
        self.data_manager.update_train_data_set_from_report_data('report.txt')
        train_sentences = self.data_manager.get_train_sentences()
        train_tags = self.data_manager.get_train_tags()
        print(len(train_sentences))
        print(len(train_tags))

        # 학습 문장으로 word2_vec_embedder 생성
        self.word2vec_embedder.make_model(train_sentences)
        #word2vec 방식으로 embedding 된 vector_set
        trainDataVecs = word2vec_embbeding_module.getAvgFeatureVecs(train_sentences , self.word2vec_embedder.model,self.word2vec_embedder.num_features)
        with open('trainDataVecs.pkl','wb') as f:
            pickle.dump(trainDataVecs, f)
        #embedding 된 vector_set과 정답지로 Sentiment_analysis 모델 생성
        self.sentiment_analysis_model.set_x_train(trainDataVecs)
        self.sentiment_analysis_model.set_y_train(train_tags)
        self.sentiment_analysis_model.make_model()

if __name__ == "__main__":
    #전처리 완료된 학습데이터 경로
    sentiment_analysis = Sentiment_analysis()
    print(sentiment_analysis.Sentiment_analysis_predict_pos_neg("테스트코드"))
