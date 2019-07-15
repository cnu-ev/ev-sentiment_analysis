from eunjeon import Mecab
import pickle

tagger = Mecab()

def tokenize(text):
    return ['/'.join(token) for token in tagger.pos(text)]
