import os
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import glove_embedder
import data_preprocessor
from lstm_attn import AttentionModel

# testttt = torch.load('index_torch.pt')
# testttt_test = torch.load('index_torch_test.pt')


class Sentiment_analysis_model:
    # def __init__(self, trainDataVecs , train_tags):
    #     self.x_train = trainDataVecs
    #     self.y_train = train_tags
    #     batch_size = 32
    #     learning_rate = 2e-5
    #     output_size = 2
    #     hidden_size = 256
    #     embedding_length = 100
    #     self.model = AttentionModel(batch_size,output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    #     loss_fn = F.cross_entropy
    #     self.train_model(self.model, testttt ,train_label )

        # self.make_model()
    def __init__(self,glove_dictionary,glove_vector):
        self.model = self.load_torch_model(glove_dictionary,glove_vector)
# print(len(x_train))

    def clip_gradient(self,clip_value):
        params = list(filter(lambda p: p.grad is not None, self.model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

    def train_model(self,trainDataVecs, y_train):
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        # self.model.cuda()
        self.model.train()
        count = int(len(trainDataVecs)/32)
        for i in range(count):
            text = trainDataVecs[32*i:32*(i+1)]
            target = y_train[32*i:32*(i+1)]
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            optim.zero_grad()
            prediction = self.model(text)
            loss = loss_fn(prediction, target)
            loss.backward()
            clip_gradient(self.model, 1e-1)
            optim.step()
            if(i%40==0):
                print(i,count)
        self.save_model()
        # text = trainDataVecs[500*count:]
        # target = y_train[500*count:]
        # target = torch.autograd.Variable(target).long()
        # optim.zero_grad()
        # prediction = model(text)
        # loss = loss_fn(prediction, target)
        # loss.backward()
        # clip_gradient(model, 1e-1)
        # optim.step()


    def eval_model(self,testDataVecs,y_test):
        self.model.eval()
        correct = 0
        count = int(len(testDataVecs)/32)
        with torch.no_grad():
            for i in range(count):
                text = testDataVecs[32*i:32*(i+1)]
                target = y_test[32*i:32*(i+1)]
                target = torch.autograd.Variable(target).long()
                if torch.cuda.is_available():
                    text = text.cuda()
                    target = target.cuda()
                prediction = self.model(text)
                loss = loss_fn(prediction, target)
                num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
                correct = correct + num_corrects
                # acc = 100.0 * num_corrects/500
                # total_epoch_loss += loss.item()
                # total_epoch_acc += acc.item()

            print('Accuracy of the network on the test images: %d %%' % (100 * correct / 75000))

    def predict_pos_neg(self,review,glove_dictionary):
        list_=data_preprocessor.indexing_embedding(review,glove_dictionary)
        list_ = torch.tensor(list_)
        self.model.batch_size = 1
        # self.model.cuda()
        text = list_
        if torch.cuda.is_available():
            text = text.cuda()
        prediction = self.model(text,batch_size=1)
        num_corrects = torch.max(prediction, 1)[1]
        return prediction,num_corrects

    def predict_pos_neg_loading_model(self,review,glove_dictionary,glove_vector):
        list_= data_preprocessor.indexing_embedding(review,glove_dictionary)
        list_ = torch.tensor(list_)
        self.model = self.load_torch_model(glove_dictionary,glove_vector)
        self.model.eval()
        self.model.batch_size = 1
        # self.model.cuda()
        text = list_
        if torch.cuda.is_available():
            text = text.cuda()
        prediction = self.model(text,batch_size=1)
        num_corrects = torch.max(prediction, 1)[1]
        return prediction,num_corrects

    def save_model(self):
        torch.save(self.model.state_dict(), 'torch_model_state_generalized_non_cuda')
        torch.save(self.model, 'torch_model_entire_generalized_non_cuda')

    def load_torch_model(self,glove_dictionary,glove_vector):
        vocab_size = len(glove_vector)
        batch_size = 32
        learning_rate = 2e-5
        output_size = 2
        hidden_size = 256
        embedding_length = 100
        word_embeddings = torch.from_numpy(glove_vector)
        model = AttentionModel(batch_size,output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
        model.load_state_dict(torch.load('torch_model_state_generalized_non_cuda'), strict=False)
        return model

    def set_x_train(trainDataVecs):
        self.x_train = trainDataVecs
    def set_y_train(train_tags):
        self.y_train = train_tags


    # predict_pos_neg("아 이규봉이 디자인한 댓글페이지 좋은데요? 추천합니다 사용할께요",run_model)
