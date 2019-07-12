import pickle
train_text_dir = './data_set_to_pickle/train_text_set.pkl''
train_text_dir = './data_set_to_pickle/test_text.pkl'
class data_manager:
    def __init__(self,train_text_dir,train_text_dir):
        with open(train_text_dir, 'rb') as f:
            self.train_text = pickle.load(f)
        with open(train_text_dir, 'rb') as f:
            self.test_text = pickle.load(f)

    def get_train_sentences(self):
        return [ sentence for sentence, tag in self.train_text]
    def get_train_tags(self):
        return [ tag for sentence, tag in self.train_text]

    def get_test_sentences(self):
        return [ sentence for sentence, tag in self.test_text]
    def get_test_tags(self):
        return [ tag for sentence, tag in self.test_text]
    
