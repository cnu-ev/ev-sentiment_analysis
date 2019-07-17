from konlpy.tag import Mecab
import pickle

tagger = Mecab()

def tokenize(text):
    return ['/'.join(token) for token in tagger.pos(text)]

class Data_preprocessor:
    def read_data(file):
        with open(file,'r',encoding='utf-8') as f:
            data = [line.split('\t') for line in f.read().splitlines()]
            data = data[1:]
        return data

if __name__ == "__main__":

    train_data = read_data('./data_set/ratings_train.txt')
    test_data = read_data('./data_set/ratings_test.txt')

    train_text = [(tokenize(row[1]),row[2]) for row in train_data]
    test_text = [(tokenize(row[1]),row[2]) for row in test_data]

    with open('./data_set_to_pickle/train_text_set.pkl', 'wb') as f:
        pickle.dump(train_text, f)

    with open('./data_set_to_pickle/test_text.pkl', 'wb') as f:
        pickle.dump(test_text, f)
