import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from cells import SimpleLSTMCell

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
import sys
import json
import time
import random
from itertools import chain
import os
import math
from model import LSTMDSSM, _START_VOCAB

random.seed(1229)

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_boolean("read_graph", False, "Set to False to build graph.")
tf.app.flags.DEFINE_integer("symbols", 400000, "vocabulary size.")
tf.app.flags.DEFINE_integer("epoch", 100, "Number of epoch.")
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("time_log_path", 'time_log.txt', "record training time")
tf.app.flags.DEFINE_integer("neg_num", 4, "negative sample number")

FLAGS = tf.app.flags.FLAGS

PAD_ID = 0
UNK_ID = 1
_START_VOCAB = ['_PAD', '_UNK']


class LSTMDSSM(object):
    """
    The LSTM-DSSM model refering to the paper: Deep Sentence Embedding Using Long Short-Term Memory Networks: Analysis and Application to Information Retrieval.
    papaer available at: https://arxiv.org/abs/1502.06922
    """

    def __init__(self,
                 num_lstm_units,
                 embed,
                 neg_num=4,
                 gradient_clip_threshold=5.0):
        self.queries = tf.placeholder(dtype=tf.string, shape=[None, None])  # shape: batch*len
        self.queries_length = tf.placeholder(dtype=tf.int32, shape=[None])  # shape: batch
        self._docs = tf.placeholder(dtype=tf.string, shape=[None, neg_num + 1, None])  # shape: batch*(neg_num + 1)*len
        self._docs_length = tf.placeholder(dtype=tf.int32, shape=[None, neg_num + 1])  # shape: batch*(neg_num + 1)
        self.docs = tf.transpose(self._docs, [1, 0, 2])  # shape: (neg_num + 1)*batch*len
        self.docs_length = tf.transpose(self._docs_length)  # shape: batch*(neg_num + 1)

        self.word2index = MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=UNK_ID,
            shared_name="in_table",
            name="in_table",
            checkpoint=True
        )

        self.learning_rate = tf.Variable(0.001, trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.momentum = tf.Variable(0.9, trainable=False, dtype=tf.float32)

        self.index_queries = self.word2index.lookup(self.queries)  # batch*len
        self.index_docs = [self.word2index.lookup(doc) for doc in tf.unstack(self.docs)]

        self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
        self.embed_queries = tf.nn.embedding_lookup(self.embed, self.index_queries)
        self.embed_docs = [tf.nn.embedding_lookup(self.embed, index_doc) for index_doc in self.index_docs]

        with tf.variable_scope('query_lstm'):
            self.cell_q = SimpleLSTMCell(num_lstm_units)
        with tf.variable_scope('doc_lstm'):
            self.cell_d = SimpleLSTMCell(num_lstm_units)

        self.states_q = dynamic_rnn(self.cell_q, self.embed_queries, self.queries_length, dtype=tf.float32,
                                         scope="simple_lstm_cell_query")[1][1]  # shape: batch*num_units
        self.states_d = [dynamic_rnn(self.cell_d, self.embed_docs[i], self.docs_length[i], dtype=tf.float32,
                                            scope="simple_lstm_cell_doc")[1][1] for i in range(neg_num + 1)]  # shape: (neg_num + 1)*batch*num_units
        self.queries_norm = tf.reduce_sum(self.states_q, axis=1)
        self.docs_norm = [tf.reduce_sum(self.states_d[i], axis=1) for i in range(neg_num + 1)]
        self.prods = [tf.reduce_sum(tf.multiply(self.states_q, self.states_d[i]), axis=1) for i in range(neg_num + 1)]
        self.sims = [(self.prods[i] / (self.queries_norm * self.docs_norm[i])) for i in range(neg_num + 1)]  # shape: (neg_num + 1)*batch
        self.sims = tf.transpose(tf.convert_to_tensor(self.sims))  # shape: batch*(neg_num + 1)
        self.gamma = tf.Variable(initial_value=1.0, expected_shape=[], dtype=tf.float32)  # scaling factor according to the paper
        self.sims = self.sims * self.gamma
        self.prob = tf.nn.softmax(self.sims)
        self.hit_prob = tf.slice(self.prob, [0, 0], [-1, 1])
        self.loss = -tf.reduce_mean(tf.log(self.hit_prob))

        self.params = tf.trainable_variables()
        opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)  # use Nesterov's method, according to the paper
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, gradient_clip_threshold)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def train_step(self, session, queries, docs):
        input_feed = {self.queries: queries['texts'],
                      self.queries_length: queries['texts_length'],
                      self._docs: docs['texts'],
                      self._docs_length: docs['texts_length']}

        output_feed = [self.loss, self.update, self.states_q, self.states_d, self.queries_norm, self.docs_norm, self.prods, self.sims, self.gamma, self.prob, self.hit_prob]
        return session.run(output_feed, input_feed)



def load_data(path, fname):
    print('Creating dataset...')
    data = []
    with open('%s/%s' % (path, fname)) as f:
        for idx, line in enumerate(f):
            line = line.strip('\n')
            tokens = line.split()
            data.append(tokens)
    return data


def build_vocab(path, data):
    print("Creating vocabulary...")
    words = set()
    for line in data:
        for word in line:
            if len(word) == 0:
                continue
            words.add(word)
    words = list(words)
    vocab_list = _START_VOCAB + words
    FLAGS.symbols = len(vocab_list)

    print("Loading word vectors...")
    embed = np.random.normal(0.0, np.sqrt(1. / (FLAGS.embed_units)), [len(vocab_list), FLAGS.embed_units])
    # debug
    # embed = np.array(embed, dtype=np.float32)
    # return vocab_list, embed
    with open(os.path.join(path, 'vector.txt')) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            info = line.split()
            if info[0] not in vocab_list:
                continue
            embed[vocab_list.index(info[0])] = [float(num) for num in info[1:]]
    embed = np.array(embed, dtype=np.float32)
    return vocab_list, embed


def gen_batch_data(data):
    def padding(sent, l):
        return sent + ['_PAD'] * (l - len(sent))

    max_len = max([len(sentence) for sentence in data])
    texts, texts_length = [], []

    for item in data:
        texts.append(padding(item, max_len))
        texts_length.append(len(item))

    batched_data = {'texts': np.array(texts), 'texts_length': np.array(texts_length, dtype=np.int32)}

    return batched_data


def train(model, sess, queries, docs):
    st, ed, loss = 0, 0, .0
    lq = len(queries) / (FLAGS.neg_num + 1)
    while ed < lq:
        st, ed = ed, ed + FLAGS.batch_size if ed + FLAGS.batch_size < lq else lq
        batch_queries = gen_batch_data(queries[st:ed])
        batch_docs = gen_batch_data(docs[st*(FLAGS.neg_num + 1):ed*(FLAGS.neg_num + 1)])
        lbq = len(batch_queries['texts'])
        batch_docs['texts'] = batch_docs['texts'].reshape(lbq, FLAGS.neg_num + 1, -1)
        batch_docs['texts_length'] = batch_docs['texts_length'].reshape(lbq, FLAGS.neg_num + 1)
        outputs = model.train_step(sess, batch_queries, batch_docs)
        if math.isnan(outputs[0]) or math.isinf(outputs[0]):
            print('nan/inf detected. ')
        loss += outputs[0]
    sess.run([model.epoch_add_op])

    return loss / len(queries)

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if FLAGS.is_train:
            print(FLAGS.__flags)
            data_queries = load_data(FLAGS.data_dir, 'queries.txt')
            data_docs = load_data(FLAGS.data_dir, 'docs.txt')
            vocab, embed = build_vocab(FLAGS.data_dir, data_queries + data_docs)

            model = LSTMDSSM(
                FLAGS.units,
                embed,
                FLAGS.neg_num)
            if FLAGS.log_parameters:
                model.print_parameters()

            if tf.train.get_checkpoint_state(FLAGS.train_dir):
                print("Reading model parameters from %s" % FLAGS.train_dir)
                model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
            else:
                print("Created model with fresh parameters.")
                tf.global_variables_initializer().run()
                op_in = model.word2index.insert(constant_op.constant(vocab),
                                                  constant_op.constant(list(range(FLAGS.symbols)), dtype=tf.int64))
                sess.run(op_in)

            summary_writer = tf.summary.FileWriter('%s/log' % FLAGS.train_dir, sess.graph)
            total_train_time = 0.0
            while model.epoch.eval() < FLAGS.epoch:
                epoch = model.epoch.eval()
                random_idxs = range(len(data_queries))
                random.shuffle(random_idxs)
                data_queries = [data_queries[i] for i in random_idxs]
                data_docs = np.reshape(data_docs, (len(data_queries), -1))
                data_docs = [data_docs[i] for i in random_idxs]
                data_docs = np.reshape(data_docs, len(data_queries) * (FLAGS.neg_num + 1))
                start_time = time.time()
                loss = train(model, sess, data_queries, data_docs)

                epoch_time = time.time() - start_time
                total_train_time += epoch_time

                summary = tf.Summary()
                summary.value.add(tag='loss/train', simple_value=loss)
                cur_lr = model.learning_rate.eval()
                summary.value.add(tag='lr/train', simple_value=cur_lr)
                summary_writer.add_summary(summary, epoch)
                model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=model.global_step)
                print("epoch %d learning rate %.10f epoch-time %.4f loss %.8f" % (
                epoch, cur_lr, epoch_time, loss))
            with open(os.path.join(FLAGS.train_dir, FLAGS.time_log_path), 'a') as fp:
                fp.writelines(['total training time: %f' % total_train_time])