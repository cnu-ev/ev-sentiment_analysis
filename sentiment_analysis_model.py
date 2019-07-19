import numpy as np

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

import word2vec_embedder
import data_preprocessor

class Sentiment_analysis_model:
    def __init__(self, trainDataVecs , train_tags):
        self.x_train = trainDataVecs
        self.y_train = np.asarray(train_tags).astype('float32')
        self.make_model()

# print(len(x_train))


# len(y_train)
    def make_model(self):
        self.run_model = models.Sequential()
        self.run_model.add(layers.Dense(64, activation='relu', input_shape=(300,)))
        self.run_model.add(layers.Dense(64, activation='relu'))
        self.run_model.add(layers.Dense(1, activation='sigmoid'))

        self.run_model.compile(optimizer=optimizers.RMSprop(lr=0.002),
                 loss=losses.binary_crossentropy,
                 metrics=[metrics.binary_accuracy])

        self.run_model.fit(self.x_train, self.y_train, epochs=10, batch_size=512)

    def model_evaluate(self,testDataVecs,test_tags):
        x_test = testDataVecs
        y_test = np.asarray(test_tags).astype('float32')
        results = self.run_model.evaluate(x_test, y_test)
        return results #[loss,score]


    def predict_pos_neg(self,review,word2vec_model,num_features):
        token = data_preprocessor.tokenize(review)
        reviewFeatureVecs = np.zeros((1,num_features),dtype="float32")
        reviewFeatureVecs[0] = review_vec = word2vec_embedder.makeFeatureVec(token, word2vec_model, num_features)
        score = float(self.run_model.predict(reviewFeatureVecs))
        return sentences, score * 100 - 50



    # predict_pos_neg("아 이규봉이 디자인한 댓글페이지 좋은데요? 추천합니다 사용할께요",run_model)
