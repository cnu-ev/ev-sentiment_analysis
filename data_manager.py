import pickle
import data_preprocessor
# train_text_dir = './data_set_to_pickle/train_text_set.pkl'
# test_text_dir = './data_set_to_pickle/test_text.pkl'

class Data_manager:
    def __init__(self,train_text_dir,test_text_dir):
        with open(train_text_dir, 'rb') as f:
            self.train_text = pickle.load(f)
        with open(test_text_dir, 'rb') as f:
            self.test_text = pickle.load(f)

    def get_train_sentences(self):
        return [ sentence for sentence, tag in self.train_text]
    def get_train_tags(self):
        return [ tag for sentence, tag in self.train_text]

    def get_test_sentences(self):
        return [ sentence for sentence, tag in self.test_text]
    def get_test_tags(self):
        return [ tag for sentence, tag in self.test_text]


    def update_data_set_from_report_data(self,report_data):
        new_data = data_preprocessor.read_data(report_data)
        new_train_data_set = [(data_preprocessor.tokenize(row[1]),row[2]) for row in train_data]
        self.train_text = self.train_text + new_train_data_set
