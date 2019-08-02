import pickle
import data_preprocessor


class Data_manager:
    preprocessed_train_text_dir = './data_set_to_pickle/preprocessed_text_train.pkl'
    preprocessed_test_text_dir = './data_set_to_pickle/preprocessed_text_test.pkl'
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
    #report_data_dir는 경로
    def update_train_data_set_from_report_data(self,report_data_dir):
        new_data = data_preprocessor.read_data(report_data_dir)
        new_train_data_set = [(data_preprocessor.tokenize(row[1]),row[2]) for row in new_data]
        self.train_text = self.train_text + new_train_data_set
        # save_data(self.train_text,self.preprocessed_train_text_dir)

    def save_data(self,data_list,data_dir):
        with open(data_dir,'wb') as f:
            pickle.dump(data_list,f)

    def load_data(self,data_dir):
        with open(data_dir, 'rb') as f:
            return pickle.load(f)
