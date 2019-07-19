from data_manager import Data_manager
from data_preprocessor import Data_preprocessor
from word2vec_embedder import Word2vec_embedder
from sentiment_analysis_model import Sentiment_analysis_model
import word2vec_embedder as word2vec_embbeding_module

# data_preprocessor = Data_preprocessor()

#data_manager 생성
data_manager = Data_manager(train_text_dir,test_text_dir)

train_sentences = data_manager.get_train_sentences()
train_tags = data_manager.get_train_tags()

#학습 문장으로 word2_vec_embedder 생성
word2vec_embedder = Word2vec_embedder(train_sentences)

#word2vec 방식으로 embedding 된 vector_set
trainDataVecs = word2vec_embbeding_module.getAvgFeatureVecs(train_sentences , word2vec_embedder.model,word2vec_embedder.num_features)

#embedding 된 vector_set과 정답지로 Sentiment_analysis 모델 생성
sentiment_analysis_model = Sentiment_analysis_model(trainDataVecs,train_tags)

#Sentiment_analysis
sentiment_analysis_model.predict_pos_neg("아따 클래스,모듈화 하기 귀찮은 것",word2vec_embedder.model,word2vec_embedder.num_features)

class Sentiment_analysis:
    def __init__(self,train_text_dir,test_text_dir):
        #전처리 완료된 학습데이터 경로

        #data_manager 생성
        self.data_manager = Data_manager(train_text_dir,test_text_dir)

        train_sentences = self.data_manager.get_train_sentences()
        train_tags = self.data_manager.get_train_tags()

        #학습 문장으로 word2_vec_embedder 생성
        self.word2vec_embedder = Word2vec_embedder(train_sentences)

        #word2vec 방식으로 embedding 된 vector_set
        trainDataVecs = word2vec_embbeding_module.getAvgFeatureVecs(train_sentences , self.word2vec_embedder.model,self.word2vec_embedder.num_features)

        #embedding 된 vector_set과 정답지로 Sentiment_analysis 모델 생성
        self.sentiment_analysis_model = Sentiment_analysis_model(trainDataVecs,train_tags)
    def Sentiment_analysis_predict_pos_neg(self,sentence):
        sentence, score = self.sentiment_analysis_model.predict_pos_neg(sentence,self.word2vec_embedder.model,self.word2vec_embedder.num_features)
        return sentence,score

if __name__ == "__main__":
    #전처리 완료된 학습데이터 경로
    train_text_dir = './data_set_to_pickle/train_text_set.pkl'
    test_text_dir = './data_set_to_pickle/test_text.pkl'

    sentiment_analysis = Sentiment_analysis(train_text_dir,test_text_dir)

    sentence, score = sentiment_analysis.Sentiment_analysis_predict_pos_neg("아 왜 이렇게 느려 짜증나게")
