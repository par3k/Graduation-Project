from flask import render_template, url_for, flash, redirect, request, abort
from App import app, db, bcrypt
from App.form import RegistrationForm, LoginForm, UpdateAccountForm, PostForm
from App.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required
from datetime import datetime

import os
import tensorflow as tf
from konlpy.tag import Okt
import gensim
import numpy as np


class Bi_LSTM():

    def __init__(self, lstm_units, num_class, keep_prob):
        self.lstm_units = lstm_units # 노드의 갯수

        with tf.compat.v1.variable_scope('forward', reuse=tf.compat.v1.AUTO_REUSE): # Left to Right side
            self.lstm_fw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_units, forget_bias=1.0, state_is_tuple=True)
            self.lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, output_keep_prob=keep_prob)

        with tf.compat.v1.variable_scope('backward', reuse=tf.compat.v1.AUTO_REUSE): # Right to Left side
            self.lstm_bw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_units, forget_bias=1.0, state_is_tuple=True)
            self.lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_bw_cell, output_keep_prob=keep_prob)

        with tf.compat.v1.variable_scope('Weights', reuse=tf.compat.v1.AUTO_REUSE): # Fully connect를 해서 softmax를 할때의 weight값
            self.W = tf.compat.v1.get_variable(name="W", shape=[2 * lstm_units, num_class], # number of class is positive and negative : 2
                                     dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.compat.v1.get_variable(name="b", shape=[num_class], dtype=tf.float32,
                                     initializer=tf.zeros_initializer())

    def logits(self, X, W, b, seq_len):
        (output_fw, output_bw), states = tf.compat.v1.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, self.lstm_bw_cell,
                                                                         dtype=tf.float32,
                                                                         inputs=X, sequence_length=seq_len) # forward cell 과 backward cell 을 합쳐서 bidirection
        ## concat fw, bw final states
        outputs = tf.compat.v1.concat([states[0][1], states[1][1]], axis=1) # Concat을 한 final state
        pred = tf.compat.v1.matmul(outputs, W) + b # final state 를 fully connected layer를 통해서 pred값으로 리턴함
        return pred

    def model_build(self, logits, labels, learning_rate=0.001):
        with tf.compat.v1.variable_scope("loss"):
            loss = tf.compat.v1.reduce_mean(
                tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))  # Softmax 를 통해서 loss 계산
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  # Adam Optimizer를 사용함

        return loss, optimizer

    def graph_build(self): # 텐서보드로 그래프를 만들때 사용하는 함수
        self.loss = tf.compat.v1.placeholderr(tf.float32)
        self.acc = tf.compat.v1.placeholder(tf.float32)
        tf.compat.v1.summary.scalar('Loss', self.loss)
        tf.compat.v1.summary.scalar('Accuracy', self.acc)
        merged = tf.compat.v1.summary.merge_all()
        return merged


class Bi_LSTM2():

    def __init__(self, lstm_units, num_class, keep_prob):
        self.lstm_units = lstm_units

        with tf.variable_scope('forward', reuse=tf.AUTO_REUSE):
            self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, forget_bias=1.0, state_is_tuple=True)
            self.lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, output_keep_prob=keep_prob)

        with tf.variable_scope('backward', reuse=tf.AUTO_REUSE):
            self.lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, forget_bias=1.0, state_is_tuple=True)
            self.lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, output_keep_prob=keep_prob)

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

    def graph_build(self, avg_loss, avg_acc):
        tf.summary.scalar('Loss', avg_loss)
        tf.summary.scalar('Accuracy', avg_acc)
        merged = tf.summary.merge_all()
        return merged


@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html')


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Pleach check again', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route("/register", methods=['GET','POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(name=form.name.data, email=form.email.data, password=hashed_password)

        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/profile", methods=['GET', 'POST'])
def profile():
    return render_template('profile.html', title='Profile')


@app.route('/sentimental')
def sentimental():
	return render_template('sentimental.html', title='Comment Sentimental Analysis')


@app.route("/predict", methods=['GET', 'POST'])
def predict():

    Batch_size = 1
    Vector_size = 300
    Maxseq_length = 95  ## Max length of training data # 이중에서 가장 긴 데이터의 문장 길이
    learning_rate = 0.001
    lstm_units = 128
    num_class = 2  # 긍정 부정 두개이기에 클래스 갯수가 2
    keep_prob = 1.0

    X = tf.compat.v1.placeholder(tf.float32, shape=[None, Maxseq_length, Vector_size], name='X')
    Y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_class], name='Y')
    seq_len = tf.compat.v1.placeholder(tf.int32, shape=[None])

    BiLSTM = Bi_LSTM(lstm_units, num_class, keep_prob)

    with tf.compat.v1.variable_scope("loss", reuse=tf.compat.v1.AUTO_REUSE):
        logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
        loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

    prediction = tf.nn.softmax(logits)

    def tokenize(doc): # 토크나이징 하는 부분
        pos_tagger = Okt()
        return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)] # 품사 테깅하는 부분

    def Convert2Vec(model_name, sentence):
        word_vec = []
        sub = []
        model = gensim.models.word2vec.Word2Vec.load(model_name)
        for word in sentence:
            if (word in model.wv.vocab):
                sub.append(model.wv[word])
            else:
                sub.append(np.random.uniform(-0.25, 0.25, 300))  ## used for OOV words
        word_vec.append(sub)
        return word_vec

    def Zero_padding(train_batch_X, Batch_size, Maxseq_length, Vector_size): # 길이를 다 같이 맞추기 위해서
                                                                                # 제일 긴 단어를 찾고 다른 단어들은 그 길이에 맞춰 0으로 빈칸을 체워주는 작업
        zero_pad = np.zeros((Batch_size, Maxseq_length, Vector_size))
        for i in range(Batch_size):
            zero_pad[i, :np.shape(train_batch_X[i])[0], :np.shape(train_batch_X[i])[1]] = train_batch_X[i]

        return zero_pad


    # modelName = "/Users/alex/Desktop/Flask Web Server/App/model/old/BiLSTM_model.ckpt" # 제작자가 만든 모델

    if request.method == 'POST':

        saver = tf.compat.v1.train.Saver()
        init = tf.compat.v1.global_variables_initializer()
        modelName = "/Users/alex/Documents/GitHub/Graduation/Flask Web Server/App/model/sentimental/BiLSTM_model_Epoch_13.ckpt"

        sess = tf.compat.v1.Session()
        sess.run(init)
        saver.restore(sess, modelName)  # 트레이닝한 BiLSTM 모델을 불러옴

        message = request.form['comment']
        print('Comment : ' + message)

        print('************************ Tokenizing Result *****************************')
        tokens = tokenize(message)
        print(tokens)

        embedding = Convert2Vec('/Users/alex/Documents/GitHub/Graduation/Flask Web Server/App/model/sentimental/Word2vec.model', tokens)
        # embedding = Convert2Vec('/Users/alex/Desktop/Flask Web Server/App/model/old/Word2vec.model', tokens) # 제작 자가 만든 모델

        zero_pad = Zero_padding(embedding, Batch_size, Maxseq_length, Vector_size)

        sess
        result = sess.run(tf.argmax(prediction, 1), feed_dict={X: zero_pad, seq_len: [len(tokens)]})
        print('************************ Negative[0] / Positive[1] - Percentage[%] ***********************')

        result_percent = sess.run((prediction), feed_dict={X: zero_pad, seq_len: [len(tokens)]})
        print(result_percent)
        print('************************ Sentimental Analysis Result ***********************')
        print(result)
    return render_template('result.html', prediction=result, title='Prediction Result')


@app.route("/post")
def post():
    posts = Post.query.all()
    return render_template('community.html', title='Post', posts=posts)


@app.route("/post/new", methods=['GET', 'POST'])
@login_required
def create_post():

    Batch_size = 1
    Vector_size = 300
    Maxseq_length = 500  # Max length of training data
    learning_rate = 0.001
    lstm_units = 128
    num_class = 4
    keep_prob = 1.0

    X = tf.compat.v1.placeholder(tf.float32, shape=[None, Maxseq_length, Vector_size], name='X')
    Y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_class], name='Y')
    seq_len = tf.compat.v1.placeholder(tf.int32, shape=[None])

    BiLSTM = Bi_LSTM2(lstm_units, num_class, keep_prob)

    with tf.compat.v1.variable_scope("loss", reuse=tf.compat.v1.AUTO_REUSE):
        logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
        loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

    prediction = tf.nn.softmax(logits)  # softmax

    def tokenize(doc): # 토크나이징 하는 부분
        pos_tagger = Okt()
        return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)] # 품사 테깅하는 부분

    def Convert2Vec(model_name, sentence):
        word_vec = []
        sub = []
        model = gensim.models.word2vec.Word2Vec.load(model_name)
        for word in sentence:
            if (word in model.wv.vocab):
                sub.append(model.wv[word])
            else:
                sub.append(np.random.uniform(-0.25, 0.25, 300))  # used for OOV words
        word_vec.append(sub)
        return word_vec

    def Zero_padding(train_batch_X, Batch_size, Maxseq_length, Vector_size): # 길이를 다 같이 맞추기 위해서
                                                                                # 제일 긴 단어를 찾고 다른 단어들은 그 길이에 맞춰 0으로 빈칸을 체워주는 작업
        zero_pad = np.zeros((Batch_size, Maxseq_length, Vector_size))
        for i in range(Batch_size):
            zero_pad[i, :np.shape(train_batch_X[i])[0], :np.shape(train_batch_X[i])[1]] = train_batch_X[i]

        return zero_pad

    form = PostForm()

    if form.validate_on_submit():

        saver2 = tf.compat.v1.train.Saver()
        init2 = tf.compat.v1.global_variables_initializer()
        modelName2 = "/Users/alex/Documents/GitHub/Graduation/Flask Web Server/App/model/category/BiLSTM.model"

        sess2 = tf.compat.v1.Session()
        sess2.run(init2)
        saver2.restore(sess2, modelName2)

        post = Post(title=form.title.data, content=form.content.data, author=current_user)
        message = form.content.data
        print('Comment : ' + message)
        print('************************ Tokenizing Result *****************************')
        tokens = tokenize(message)
        print(tokens)

        embedding2 = Convert2Vec('/Users/alex/Documents/GitHub/Graduation/Flask Web Server/App/model/category/league_category.embedding', tokens)
        zero_pad = Zero_padding(embedding2, Batch_size, Maxseq_length, Vector_size)
        sess2
        result2 = sess2.run(prediction, feed_dict={X: zero_pad, seq_len: [len(tokens)]})  # tf.argmax(prediction, 1)이 여러 prediction 값중 max 값 1개만 가져옴

        point = result2.ravel().tolist()
        Tag = ["EPL(England) :", "Laliga (Spain) :", "Bundesliga (Germany) :", "Seria (Italy) :"]
        print('************************ Category classification Result ***********************')
        for t, i in zip(Tag, point):
            print(t, round(i * 100, 2), "%")

        db.session.add(post)
        db.session.commit()
        flash('Your post has been created!', 'success')
        return redirect(url_for('post'))
    return render_template('post.html', title='Create post', form=form)


@app.route("/post/<int:post_id>")
def post_ind(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', title=post.title, post=post)