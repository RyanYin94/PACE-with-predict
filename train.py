# coding=utf-8
from collections import defaultdict

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
import pickle
from time import time
import dataset
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def init_normal(shape, name=None):
    return initializers.normal(shape)


# model = get_Model(num_users, num_items, 10, 37002, 12223, [16,8], [0, 0])
def get_Model(num_users, num_items, latent_dim, user_con_len, item_con_len, layers=[20, 10, 5], regs=[0, 0, 0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                               embeddings_initializer='uniform', W_regularizer=l2(regs[0]), input_length=1)
    item_embedding = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                               embeddings_initializer='uniform', W_regularizer=l2(regs[1]), input_length=1)

    user_latent = Flatten()(user_embedding(user_input))
    item_latent = Flatten()(item_embedding(item_input))

    vector = merge([user_latent, item_latent], mode='concat')

    for i in range(len(layers)):
        hidden = Dense(layers[i], activation='relu', init='lecun_uniform', name='ui_hidden_' + str(i))
        vector = hidden(vector)

    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(vector)

    user_context = Dense(user_con_len, activation='sigmoid', init='lecun_uniform', name='user_context')(user_latent)
    item_context = Dense(item_con_len, activation='sigmoid', init='lecun_uniform', name='item_context')(item_latent)

    model = Model(input=[user_input, item_input], output=[prediction, user_context, item_context])
    return model


# model = get_Model(100000, 100000, 10, 37002, 12223)
# config = model.get_config()
# weights = model.get_weights()


def get_train_instances(train_data):
    while 1:
        user_input = train_data['user_input']
        item_input = train_data['item_input']
        ui_label = train_data['ui_label']
        u_context = train_data['u_context']
        s_context = train_data['s_context']
        for i in range(len(u_context)):
            u = []
            it = []
            p = []
            u.append(user_input[i])
            it.append(item_input[i])
            p.append(ui_label[i])
            x = {'user_input': np.array(u), 'item_input': np.array(it)}
            y = {'prediction': np.array(p), 'user_context': np.array(u_context[i]).reshape((1, 1384)),
                 'item_context': np.array(s_context[i]).reshape((1, 871))}
            yield (x, y)


train = None
with open('data/traindata_small.pkl', 'rb') as f1:
    train = pickle.load(f1)

test = None
with open('data/testdata_small.pkl', 'rb') as f2:
    test = pickle.load(f2)

user_input = train['user']
item_input = train['spot']
ui_label = train['label']

user_exist_unique = sorted(list(set(user_input)))

dict_u_p = defaultdict(set)
for i in range(len(user_input)):
    dict_u_p[user_input[i]].add(item_input[i])


item_input_test = range(871)
ground_truth_user = test['user']
ground_truth_item = test['spot']
dict_gt = defaultdict(set)
for i in range(len(ground_truth_user)):
    dict_gt[ground_truth_user[i]].add(ground_truth_item[i])

data = dataset.Dataset('_small')
data.generateContextLabels()
contexts = data.context_data
u_context, s_context = contexts['user_context'], contexts['spot_context']
train_data = {}
train_data['user_input'] = user_input
train_data['item_input'] = item_input
train_data['ui_label'] = ui_label
train_data['u_context'] = u_context
train_data['s_context'] = s_context


# test_data = {}
# test_data['user_input'] = user_input_test
# test_data['item_input'] = item_input_test
# test_data['ui_label'] = ui_label_test


def ndcg_at_k(r, k):
    # dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    # if not dcg_max:
    #     return 0.
    # return dcg_at_k(r, k) / dcg_max
    idcg = 1.0
    dcg = float(r[0])
    for i, p in enumerate(r[1:k]):
        if p == 1:
            dcg += 1.0 / np.log(i + 2)
        idcg += 1.0 / np.log(i + 2)
    return dcg / idcg


def map_at_k(r, k, fm):
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(r):
        if p == 1:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(fm, k)


if __name__ == '__main__':
    layers = eval("[16,8]")
    reg_layers = eval("[0,0]")
    learner = "Adam"
    learning_rate = 0.0001
    epochs = 100
    batch_size = 1024
    verbose = 1
    losses = ['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy']

    num_users, num_items = len(user_input), len(item_input)
    num_user_context = len(u_context[0])
    num_item_context = len(s_context[0])

    print('Build model')


    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.accs = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.accs.append(logs.get('acc'))


    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0)

    history = LossHistory()
    model = get_Model(num_users, num_items, 10, 1384, 871, layers, reg_layers)

    model.compile(optimizer=Adam(lr=learning_rate), loss=losses, metrics=['accuracy'])

    print('Start Training')

    for epoch in range(epochs):
        t1 = time()
        hist = model.fit_generator(get_train_instances(train_data), samples_per_epoch=batch_size, nb_epoch=10,
                                   verbose=1, callbacks=[history, board])

    p_5 = []
    p_10 = []
    p_20 = []
    p_50 = []
    r_5 = []
    r_10 = []
    r_20 = []
    r_50 = []
    map_5 = []
    map_10 = []
    map_20 = []
    map_50 = []
    ndcg_5 = []
    ndcg_10 = []
    ndcg_20 = []
    ndcg_50 = []

    for unique_user in user_exist_unique:
        user_input_test = [unique_user] * 871
        test_input = {'user_input': np.array(user_input_test), 'item_input': np.array(item_input_test)}
        pred = model.predict(test_input)[0]
        all_items = set(range(871))

        rating = pred
        # user = test_input['user_input'][1]

        test_items = list(all_items - dict_u_p[unique_user])
        item_score = []
        for j in test_items:
            item_score.append((j, rating[j]))

        item_score = sorted(item_score, key=lambda x: x[1], reverse=True)  # 把评分从大到小排序
        item_sort = [x[0] for x in item_score]  # 把排序好的评分转化成对应的poi

        r = []
        for i in item_sort:
            if i in dict_gt[unique_user]:
                r.append(1)  # 有多少1就是预测对多少个poi
            else:
                r.append(0)

        p_5.append(np.mean(r[:5]))
        p_10.append(np.mean(r[:10]))
        p_20.append(np.mean(r[:20]))
        p_50.append(np.mean(r[:50]))
        fm = float(len(dict_gt[unique_user]))
        if fm == 0:
            r_5.append(0)
            r_10.append(0)
            r_20.append(0)
            r_50.append(0)

            map_5.append(0)
            map_10.append(0)
            map_20.append(0)
            map_50.append(0)
        else:
            r_5.append(np.sum(r[:5]) / fm)
            r_10.append(np.sum(r[:10]) / fm)
            r_20.append(np.sum(r[:20]) / fm)
            r_50.append(np.sum(r[:50]) / fm)

            map_5.append(map_at_k(r, 5, fm))
            map_10.append(map_at_k(r, 10, fm))
            map_20.append(map_at_k(r, 20, fm))
            map_50.append(map_at_k(r, 50, fm))

        ndcg_5.append(ndcg_at_k(r, 5))
        ndcg_10.append(ndcg_at_k(r, 10))
        ndcg_20.append(ndcg_at_k(r, 20))
        ndcg_50.append(ndcg_at_k(r, 50))

        print "best P@5: ", np.mean(p_5)
        print "best P@10: ", np.mean(p_10)
        print "best P@20: ", np.mean(p_20)
        print "best P@50: ", np.mean(p_50)
        print "best R@5: ", np.mean(r_5)
        print "best R@10: ", np.mean(r_10)
        print "best R@20: ", np.mean(r_20)
        print "best R@50: ", np.mean(r_50)
        print "best M@5: ", np.mean(map_5)
        print "best M@10: ", np.mean(map_10)
        print "best M@20: ", np.mean(map_20)
        print "best M@50: ", np.mean(map_50)
        print "best D@5: ", np.mean(ndcg_5)
        print "best D@10: ", np.mean(ndcg_10)
        print "best D@20: ", np.mean(ndcg_20)
        print "best D@50: ", np.mean(ndcg_50)
        print "**************************************"
