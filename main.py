from data_manager import Data_manager
import data_preprocessor
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
        self.sentiment_analysis_model.set_x_train(trainDataVecs)
        self.sentiment_analysis_model.set_y_train(train_tags)
        self.sentiment_analysis_model.make_model()


if __name__ == "__main__":
    #전처리 완료된 학습데이터 경로
    sentiment_analysis = Sentiment_analysis()
    test_data = data_preprocessor.read_data('text_test.txt')
    test_data[50001]
    test_text = [ x[1] for x in test_data ]
    test_label_ = [ int(x[2]) for x in test_data ]

    len(test_text)
    len(test_label_)

    test_text[50001]
    test_label_[50001]

    with open('testDataVecs.pkl', 'rb') as f:
        testDataVecs=pickle.load(f)

    train_sentences = sentiment_analysis.data_manager.get_train_sentences()
    trainDataVecs = word2vec_embbeding_module.getAvgFeatureVecs(train_sentences , sentiment_analysis.word2vec_embedder.model,sentiment_analysis.word2vec_embedder.num_features)
    test_sentences = sentiment_analysis.data_manager.get_test_sentences()
    test_label = sentiment_analysis.data_manager.get_test_tags()
    testDataVecs = word2vec_embbeding_module.getAvgFeatureVecs(test_sentences , sentiment_analysis.word2vec_embedder.model,sentiment_analysis.word2vec_embedder.num_features)

    len(train_sentences)
    len(test_sentences)
    test_sentences[50000]
    test_label = [int(x) for x in test_label]
    # test_sentences[50000:50010]
    with open('trainDataVecs.pkl','wb') as f:
        pickle.dump(trainDataVecs,f)
    with open('testDataVecs.pkl','wb') as f:
        pickle.dump(testDataVecs,f)

    len(testDataVecs)


    sentiment_analysis.sentiment_analysis_model.model_evaluate(testDataVecs,test_label)
    sentiment_analysis.Sentiment_analysis_predict_pos_neg(test_text[1])[1]
    data_manager = Data_manager()
    type(test_label[0])
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(test_text)):
        result = sentiment_analysis.Sentiment_analysis_predict_pos_neg(test_text[i])[1]
        if( test_label_[i] == 1 and result >= 0):
            TP = TP + 1
        if( test_label_[i] == 1 and result < 0):
            FN = FN + 1
        if( test_label_[i] == 0 and result >= 0):
            FP = FP + 1
        if( test_label_[i] == 0 and result < 0):
            TN = TN + 1


    TP
    FP
    FN
    TN
