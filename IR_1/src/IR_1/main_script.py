from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from IR_1 import network

def main(_): 
    
    if FLAGS.model=='logreg':
        network.run_net(FLAGS)
#     elif FLAGS.model=='qA4':
#         qA4.run_net(1e-3, FLAGS)
    
   
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('-saved_model_dir', type=str, default='C:/Users/Dave/Documents/GI13-Advanced/Assignment2/model',
                        help='Directory where trained models are saved')
    parser.add_argument('-lr', type=str, default='1e-4', help='learning rate')
    parser.add_argument('-sm', type=str, default=None, help='saved model to evaluate')
    parser.add_argument('-game', type=str, default=None, help='game to evaluate')
    parser.add_argument('-nhidden', type=str, default=None, help='number hidden units')
    parser.add_argument('-eval', action='store_true', help='just evaluate with saved model')
    parser.add_argument('-bn', action='store_true', help='use batch normalization')
    parser.add_argument('--model', type=str, default='P1_a', 
        help='which model to run, one of [P1_a, P1_b, P1_c, P1_d, P2_a, P2_b, P3_c, P4_d]')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
