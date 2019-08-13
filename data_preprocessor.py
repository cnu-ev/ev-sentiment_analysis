from eunjeon import Mecab
# from konlpy.tag import Mecab
import pickle
from glove import Glove

tagger = Mecab()

def tokenize(text):
    return ['/'.join(token) for token in tagger.pos(text)]


def read_data(file):
    with open(file,'r',encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

def texts_generalizing(texts):
    sentence = tagger.pos(texts)
    for word in sentence:
        word_,tag = word
        if('JK' in tag or 'JC' in tag or 'JX' in tag or 'VC' in tag):
            sentence.remove(word)
    return sentence


def indexing_embedding(texts,glove_dictionary):
    sentence=texts
    zero_vecotr_index = len(glove_dictionary)
    list_ = []
    for word in sentence:
        if(word in glove_dictionary):
            list_.append(glove_dictionary[word])
        else:
            list_.append(zero_vecotr_index)
    count = 200-len(list_)
    while(count):
        list_.append(zero_vecotr_index)
        count = count - 1
    return list_

if __name__ == "__main__":

    train_data = read_data('./data_set/ratings_train.txt')
    test_data = read_data('./data_set/ratings_test.txt')

    train_text = [(tokenize(row[1]),row[2]) for row in train_data]
    test_text = [(tokenize(row[1]),row[2]) for row in test_data]

    with open('./data_set_to_pickle/train_text_set.pkl', 'wb') as f:
        pickle.dump(train_text, f)

    with open('./data_set_to_pickle/test_text.pkl', 'wb') as f:
        pickle.dump(test_text, f)
