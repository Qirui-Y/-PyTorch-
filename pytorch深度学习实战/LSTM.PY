# coding:utf-8
from __future__ import print_function
import numpy as np
import os
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import tensorflow
texts = [] # list of text samples
labels_index = {} # dictionary mapping label name to numeric id
labels = [] # list of label ids
TEXT_DATA_DIR='数据集/20_newsgroup'
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                f = open(fpath,'rb')
                texts.append(f.read())
                f.close()
                labels.append(label_id)
print('Found %s texts.' % len(texts))
print('labels length %s .' % len(labels))

MAX_NB_WORDS=20000
EMBEDDING_DIM=100
HIDDEN_DIM=100
MAX_SEQUENCE_LENGTH=400
epochs = 2
batch_size = 1

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print('word_index:',word_index)
data=[]
for i in range(len(sequences)):
    data.append(np.asarray(sequences[i]))

labels =np.asarray(labels)
print("data len:",len(data))
#print("data sample:",data[0])

print('Shape of data tensor:', len(data))
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
VALIDATION_SPLIT=0.2
nb_validation_samples = int(VALIDATION_SPLIT * len(data))

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]
print("train length:",len(x_train))
print("test length:",len(x_test))

target_size= len(labels_index)
num_samples=len(x_train)
'''
build torch model
'''
class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.word_embeddings = nn.Embedding(MAX_NB_WORDS, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,2,batch_first =True)
        self.hidden2tag = nn.Linear(self.hidden_dim, 20)
        self.hidden = self.init_hidden()
        self.drop = nn.Dropout(p=0.2)
    def init_hidden(self):
        return (Variable(torch.zeros(2, batch_size, self.hidden_dim)), Variable(torch.zeros(2, batch_size, self.hidden_dim)))
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = self.drop(embeds)
        #lstm_out, self.hidden = self.lstm(embeds,self.hidden)
        lstm_out= self.lstm(embeds)
        out = lstm_out[0][:,-1,:]
        flat = out.view(-1, HIDDEN_DIM)
        tag_space = self.hidden2tag(flat)
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

model = LSTMNet()
#if os.path.exists('torch_lstm.pkl'):
# model = torch.load('torch_lstm.pkl')
print(model)
'''
trainning
'''
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#loss=torch.nn.CrossEntropyLoss(size_average=True)
def train(epoch,x_train,y_train):
    num_batchs = num_samples/ batch_size
    model.train()
    model.hidden = model.init_hidden()
    for k in range(num_batchs):
        start,end = k*batch_size,(k+1)*batch_size
        data=Variable( torch.Tensor(x_train[start:end]).long())
        target = Variable(torch.Tensor(y_train[start:end]).long(),requires_grad=False)
        #embeds = word_embeddings( Variable(t)) #,requires_grad=False)) 
        #requires_grad=False)a, target = Variable(x_train[start:end],requires_grad=False), Variable(y_train[start:end],requires_grad=False)
        #data, target = Variable(x_train[start:end]), Variable(y_train[start:end])
        optimizer.zero_grad()
        #print("train data size:",data.size())
        output = model(data)
        #print("output :",output.size())
        #print("target :",target.size())
        loss = F.nll_loss(output,target) #criterion(output,target)
        loss.backward()
        optimizer.step()
        if k % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, k * len(data), num_samples,100. * k / num_samples, loss.data[0]))
    torch.save(model, 'torch_lstm.pkl')
'''
evaludate
'''
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    len=400
    print("x_test size:",len(x_test))
    for i in range(len):
        data, target = Variable(torch.Tensor(x_test[i:i+1]).long()), Variable(torch.Tensor(y_test[i:i+1]).long(),requires_grad=False)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        test_loss = test_loss
        test_loss /= len(x_test) # loss function already averages over batch size
        if i % 10 == 0:
            print("single loss:",test_loss,",right counts:",correct)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(x_test),100. * correct / len))

for epoch in range(1,epochs):
    train(epoch,x_train,y_train)
    test(epoch)










