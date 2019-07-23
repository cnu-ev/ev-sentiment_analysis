import pickle
import data_preprocessor


class Data_manager:
    preprocessed_train_text_dir = './data_set_to_pickle/train_text_set.pkl'
    preprocessed_test_text_dir = './data_set_to_pickle/test_text.pkl'
    def __init__(self):
        self.train_text = self.load_data(self.preprocessed_train_text_dir)
        self.test_text = self.load_data(self.preprocessed_test_text_dir)

    def get_train_sentences(self):
        return [ sentence for sentence, tag in self.train_text]
    def get_train_tags(self):
        return [ tag for sentence, tag in self.train_text]

    def get_test_sentences(self):
        return [ sentence for sentence, tag in self.test_text]
    def get_test_tags(self):
        return [ tag for sentence, tag in self.test_text]
