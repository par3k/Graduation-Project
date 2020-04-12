# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:15:44 2018

@author: jbk48
"""

import os
import tensorflow as tf

import Word2Vec
import gensim
import numpy as np


class Bi_LSTM():

    def __init__(self, lstm_units, num_class, keep_prob):
        self.lstm_units = lstm_units

        with tf.variable_scope('forward', reuse=tf.AUTO_REUSE):
            self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, forget_bias=1.0, state_is_tuple=True)
            self.lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, output_keep_prob=keep_prob)

        with tf.variable_scope('backward', reuse=tf.AUTO_REUSE):
            self.lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, forget_bias=1.0, state_is_tuple=True)
            self.lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_bw_cell, output_keep_prob=keep_prob)

        with tf.variable_scope('Weights', reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable(name="W", shape=[2 * lstm_units, num_class],
                                     dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable(name="b", shape=[num_class], dtype=tf.float32,
                                     initializer=tf.zeros_initializer())

    def logits(self, X, W, b, seq_len):
        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, self.lstm_bw_cell,
                                                                         dtype=tf.float32,
                                                                         inputs=X, sequence_length=seq_len)
        ## concat fw, bw final states
        outputs = tf.concat([states[0][1], states[1][1]], axis=1)
        pred = tf.matmul(outputs, W) + b
        return pred

    def model_build(self, logits, labels, learning_rate=0.001):
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))  # Softmax loss
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  # Adam Optimizer

        return loss, optimizer

    def graph_build(self):
        self.loss = tf.placeholder(tf.float32)
        self.acc = tf.placeholder(tf.float32)
        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('Accuracy', self.acc)
        merged = tf.summary.merge_all()
        return merged


W2V = Word2Vec.Word2Vec()

Batch_size = 1
Vector_size = 300
Maxseq_length = 95   ## Max length of training data
learning_rate = 0.001
lstm_units = 128
num_class = 2
keep_prob = 1.0

X = tf.placeholder(tf.float32, shape = [None, Maxseq_length, Vector_size], name = 'X')
Y = tf.placeholder(tf.float32, shape = [None, num_class], name = 'Y')
seq_len = tf.placeholder(tf.int32, shape = [None])

BiLSTM = Bi_LSTM(lstm_units, num_class, keep_prob)

with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
    loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

prediction = tf.nn.softmax(logits)


def Convert2Vec(model_name, sentence):
    
    word_vec = []
    sub = []
    model = gensim.models.word2vec.Word2Vec.load(model_name)
    for word in sentence:
        if(word in model.wv.vocab):
            sub.append(model.wv[word])
        else:
            sub.append(np.random.uniform(-0.25,0.25,300)) ## used for OOV words
    word_vec.append(sub)
    return word_vec



saver = tf.train.Saver()
init = tf.global_variables_initializer()
modelName = "./BiLSTM_model.ckpt"

sess = tf.Session()
sess.run(init)
saver.restore(sess, modelName)

os.chdir("..")

def Grade(sentence):

    tokens = W2V.tokenize(sentence)
    
    embedding = Convert2Vec('./Word2Vec/Word2vec.model', tokens)
    zero_pad = W2V.Zero_padding(embedding, Batch_size, Maxseq_length, Vector_size)
    global sess
    result =  sess.run(tf.argmax(prediction,1), feed_dict = {X: zero_pad , seq_len: [len(tokens)] } ) 
    if(result == 1):
        print("Positive Sentence")
    else:
        print("Negative Sentence")
            
while(1):
    s = input("문장을 입력하세요 : ")
    if(s == str(1)):
        break
    else:
        Grade(s)
    
