from data_manager import Data_manager
from data_preprocessor import Data_preprocessor
from word2vec_embedder import Word2vec_embedder
from sentiment_analysis_model import Sentiment_analysis_model


# data_preprocessor = Data_preprocessor()

#전처리 완료된 학습데이터 경로
train_text_dir = './data_set_to_pickle/train_text_set.pkl'
test_text_dir = './data_set_to_pickle/test_text.pkl'


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
