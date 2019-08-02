from data_manager import Data_manager
# from data_preprocessor import Data_preprocessor
from word2vec_embedder import Word2vec_embedder
from sentiment_analysis_model import Sentiment_analysis_model
import word2vec_embedder as word2vec_embbeding_module
import pickle

class Sentiment_analysis:
    def __init__(self):
        #data_manager 생성
        self.data_manager = Data_manager()

        train_sentences = self.data_manager.get_train_sentences()
        train_tags = self.data_manager.get_train_tags()

        #학습 문장으로 word2_vec_embedder 생성
        self.word2vec_embedder = Word2vec_embedder(train_sentences)

        #word2vec 방식으로 embedding 된 vector_set
        with open('trainDataVecs.pkl', 'rb') as f:
            trainDataVecs = pickle.load(f)

        #embedding 된 vector_set과 정답지로 Sentiment_analysis 모델 생성
        self.sentiment_analysis_model = Sentiment_analysis_model(trainDataVecs,train_tags)

    def Sentiment_analysis_predict_pos_neg(self,sentence):
        # sentence, score = self.sentiment_analysis_model.predict_pos_neg_by_loaded_model(sentence,self.word2vec_embedder.model,self.word2vec_embedder.num_features)
        sentence, score = self.sentiment_analysis_model.predict_pos_neg(sentence,self.word2vec_embedder.model,self.word2vec_embedder.num_features)

        return sentence,score

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
        self.sentiment_analysis_model = Sentiment_analysis_model(trainDataVecs,train_tags)


if __name__ == "__main__":

    sentiment_analysis = Sentiment_analysis()
    sentiment_analysis.Sentiment_analysis_predict_pos_neg("")
