import os, sys
import tensorflow as tf
from cosine import Cosine
from model_param import ModelParam

class Att_BiLSTM(object):

    def __init__(self):
        self.model_param = ModelParam()

    def process_features(self, input_x):

        used_features = input_x
        
        with tf.variable_scope("lstm"):
            lstm_forward = tf.nn.rnn_cell.LSTMCell(num_units=self.model_param.lstm_hidden, name='lstm_forward')
            lstm_backward = tf.nn.rnn_cell.LSTMCell(num_units=self.model_param.lstm_hidden, name='lstm_backward')
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_forward, cell_bw=lstm_backward, inputs=used_features, dtype=tf.float32)
            lstm_output = tf.concat([output_fw, output_bw], axis=2)
            lstm_output_rsp = tf.reshape(lstm_output, [-1, 2 * self.model_param.lstm_hidden])

        with tf.variable_scope("self-attention"):
            W_s1 = tf.get_variable(name='W_s1', shape=[2*self.model_param.lstm_hidden, self.model_param.attention_hidden])
            H_s1 = tf.nn.tanh(tf.matmul(lstm_output_rsp, W_s1))
            W_s2 = tf.get_variable(name='W_s2', shape=[self.model_param.attention_hidden, self.model_param.attention_num])
            H_s2 = Cosine(H_s1, W_s2)
            H_s2_rsp = tf.transpose(tf.reshape(H_s2, [-1, self.model_param.seq_length, self.model_param.attention_num]), [0, 2, 1])
            A = tf.nn.softmax(logits=H_s2_rsp, axis=-1, name="attention")
            self.heat_matrix = A
            M = tf.matmul(A, lstm_output)
            M_flat = tf.reshape(M, [-1, 2 * self.model_param.lstm_hidden * self.model_param.attention_num])

        with tf.variable_scope("penalize"):
            AA_T = tf.matmul(A, tf.transpose(A, [0, 2, 1]))
            I = tf.eye(self.model_param.attention_num, batch_shape=[tf.shape(A)[0]])
            P = tf.square(tf.norm(AA_T - I, axis=[-2, -1], ord="fro"))
            loss_P = tf.reduce_mean(self.model_param.penalty_C * P)

        with tf.variable_scope("dense"):
            fc = tf.layers.dense(inputs=M_flat, units=self.model_param.fc1_dim, activation=None, name='fc1')
            fc = tf.nn.relu(fc)
            fc = tf.layers.dense(inputs=fc, units=self.model_param.output_dim, activation=None, name='fc_out')

        return fc, loss_P





