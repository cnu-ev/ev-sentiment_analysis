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


# print(len(x_train))


# len(y_train)
    def make_model(self):
        run_model = models.Sequential()
        run_model.add(layers.Dense(64, activation='relu', input_shape=(300,)))
        run_model.add(layers.Dense(64, activation='relu'))
        run_model.add(layers.Dense(1, activation='sigmoid'))

        run_model.compile(optimizer=optimizers.RMSprop(lr=0.002),
                 loss=losses.binary_crossentropy,
                 metrics=[metrics.binary_accuracy])

        run_model.fit(x_train, y_train, epochs=10, batch_size=512)

    def model_evaluate(self,testDataVecs,test_tags):
        x_test = testDataVecs
        y_test = np.asarray(test_tags).astype('float32')
        results = run_model.evaluate(x_test, y_test)
        return results #[loss,score]
