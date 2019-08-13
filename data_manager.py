import pickle
import data_preprocessor


class Data_manager:
    generalized_train_data_tagged_dir = './data_set_to_pickle/generalized_train_data_tagged.pkl'
    generalized_test_data_tagged_dir = './data_set_to_pickle/generalized_test_data_tagged.pkl'
    train_label_dir = 'data_set_to_pickle/train_label.pkl'
    test_label_dir = 'data_set_to_pickle/test_label.pkl'

    def __init__(self):
        self.train_text = self.load_data(self.generalized_train_data_tagged_dir)
        self.test_text = self.load_data(self.generalized_test_data_tagged_dir)
        self.train_label = self.load_data(self.train_label_dir)
        self.test_label = self.load_data(self.test_label_dir)

    def get_train_sentences(self):
        return self.train_text
    def get_train_tags(self):
        return self.train_label

    def get_test_sentences(self):
        return self.test_text
    def get_test_tags(self):
        return self.test_label

    #report_data_dir는 경로
    def update_train_data_set_from_report_data(self,report_data_dir):
        new_data = data_preprocessor.read_data(report_data_dir)
        new_train_data_text = [(data_preprocessor.texts_generalizing(row[1])) for row in new_data]
        new_train_data_label = [ row[2] for row in new_data ]
        self.train_text = self.train_text + new_train_data_text
        self.train_label = self.train_label + new_train_data_label
        # save_data(self.train_text,self.preprocessed_train_text_dir)

    def save_data(self,data_list,data_dir):
        with open(data_dir,'wb') as f:
            pickle.dump(data_list,f)

    def load_data(self,data_dir):
        with open(data_dir, 'rb') as f:
            return pickle.load(f)
