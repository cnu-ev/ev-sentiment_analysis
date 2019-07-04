from eunjeon import Mecab
import pickle

def read_data(file):
    with open(file,'r',encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data


def tokenize(text):
    return ['/'.join(token) for token in tagger.pos(text)]
