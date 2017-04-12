import gym
import os
import numpy as np
import random
import tensorflow as tf
import pickle as pi
import matplotlib.pyplot as plt
import datetime#, time
import sys
import collections

from timeit import default_timer as timer
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config
from tensorflow.core.framework import summary_pb2


NUM_FEATURES = 136
NUM_RELEVANCE_LEVELS = 5

def save_model(session, model_name, root_dir):
    if not os.path.exists(root_dir + '/model/'):
        os.mkdir(root_dir + '/model/')
    saver = tf.train.Saver(write_version=1)
    saver.save(session, root_dir + '/model/' + model_name +'.ckpt')


        
def weight_variable(shape):
#     initial = tf.constant(0.0, shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, 'W')

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
#     initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, 'b')
 
def build_net(n_inputs, n_outputs):
    
    x = tf.placeholder(tf.float32, [None, n_inputs], 'x')
    
    W_1 = weight_variable([n_inputs, n_outputs])
    
#     h_1 = tf.nn.relu(tf.matmul(x, W_1))
    
    logits = tf.matmul(x, W_1)

    y = tf.nn.sigmoid(logits)


#     l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
#     total_loss = loss + LAMBDA * l2_loss
#     tf.summary.scalar('residual', residual)

    return x, y

def build_net_softmax(n_inputs, n_outputs, layer_sizes, keep_prob, lambda_val):
    
    x = tf.placeholder(tf.float32, [None, n_inputs], 'x')
    previous_layer = x
    
    for n_units in layer_sizes:  
        W = weight_variable([n_inputs, n_units])
        b = bias_variable([n_units])    
        lin = tf.matmul(previous_layer, W) + b  
        h = tf.nn.relu(lin)
      
        h_drop = tf.nn.dropout(h, keep_prob) 
        n_inputs = n_units 
        previous_layer = h_drop

    W = weight_variable([n_inputs, n_outputs])
    b = bias_variable([n_outputs])
    logits = tf.matmul(previous_layer, W) + b
    
    y = tf.nn.sigmoid(logits, name='sigmoid_outputs')



    targets = tf.placeholder(tf.int32, [None], 'targets') # 0 or 1
    Xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
    loss = tf.reduce_mean(Xent_loss)
    
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    total_loss = loss + lambda_val * l2_loss

    return x, y, total_loss, targets

def has_constant_relevance(rfps):
    relevance = rfps[0].relevance
    for rfp in rfps:
        if rfp.relevance != relevance:
            return False
    return True
    
class DataManager:

    def __init__(self, dic):
        self.dict = dic
        return
        
    def sample(self, batch_size, n_inputs, use_simple_logreg, use_softmax):
        
        inputs = np.zeros((batch_size, n_inputs))
        targets = np.zeros((batch_size, 1))
        for i in range(batch_size):
            qid = random.sample(list(self.dict), 1)[0]
            rfps = self.dict[qid]
            rfp1 = random.sample(rfps, 1)[0]

            if use_simple_logreg:
                while True:
                    rfp2 = random.sample(rfps, 1)[0]
                    if rfp2.relevance != rfp1.relevance:
                        break
                
                if random.random() > 0.5:
                    rfp1_rnd = rfp2
                    rfp2_rnd = rfp1
                else:
                    rfp1_rnd = rfp1
                    rfp2_rnd = rfp2
                
                inputs[i,:NUM_FEATURES] = rfp1_rnd.features
                inputs[i,NUM_FEATURES:] = rfp2_rnd.features
                if rfp1_rnd.relevance > rfp2_rnd.relevance:
                    targets[i,:] = 1
                else:
                    targets[i,:] = 0
                    
            elif use_softmax:
                inputs[i,:] = rfp1.features
#                 targets[i,:] = np.zeros((NUM_RELEVANCE_LEVELS))
#                 targets[i,rfp1.relevance] = 1
                targets[i,0] = rfp1.relevance
                
                
        return inputs, targets
   
class Query_URL_pair:
    
    def __init__(self, tokens):
        self.tokens = tokens
        self.relevance = None
        self.qid = None
        self.features = []
        self.parse()
        return
    
    def parse(self):
        self.relevance = int(self.tokens[0])
        self.qid = int(self.tokens[1].split(':')[1])
        for t in self.tokens[2:]:
            self.features.append(float(t.split(':')[1]))
        return 
         
class Relevance_feature_pair:
    
    def __init__(self, relevance, features):
        self.relevance = relevance
        self.features = features
        return

def arrange_by_qid(data):
    
    dic = collections.defaultdict(list)

    for d in data:
        rfps = dic[d.qid]
        rfps.append( Relevance_feature_pair(d.relevance, d.features) )
        
    count = 0
    key_list = list(dic.keys())
    for qid in key_list:
        rfps = dic[qid]
        if(has_constant_relevance(rfps)):
            del dic[qid]
            count += 1
    print('deleted ' + str(count) + ' entries out of ' + str(len(key_list)))
                
    return dic
    
def normalize(data, use_saved):
    n = len(data)
    n_inputs = len(data[0].features)
    features = np.zeros((n, n_inputs))
    
    for i, d in enumerate(data):
        features[i,:] = d.features

    if not use_saved:
        med = np.median(features, axis=0)
        diff = features - med
        ab = np.abs(diff)
        std = np.median(ab, axis=0)
        std[std == 0] = 1

        with open('median_std.pkl', 'wb') as f:
            pi.dump((med, std), f)  
    else:   
        with open('median_std.pkl', 'rb') as f:
            med, std = pi.load(f)  
        
    features_norm = (features - med)/std
    
    for i, d in enumerate(data):
        d.features = features_norm[i,:]
        
    return data

def import_data(use_binary, fold, data_type, num_data_points=100):
    data = []
    fn = 'Fold' + str(fold) + '/' + data_type
    
    if not use_binary:
        with open(fn+'.txt') as f:
            for line in f:
    #             print(line)
                q_u_pair = Query_URL_pair( line.split() )
                data.append(q_u_pair)
#                 if len(data) > 10000:
#                     break;

        normalize(data, use_saved=True)
        dic = arrange_by_qid(data)
        
        with open(fn+'.pkl', 'wb') as f:
            pi.dump(dic, f)
            
        l_dic = list(dic)
        random.shuffle(l_dic)
        for n in [100,500,1000]:
            with open(fn+'_'+str(n)+'.pkl', 'wb') as f:
                keys = l_dic[:n]
                subdict={a:dic[a] for a in keys}
                pi.dump(subdict, f)
        exit()
    else:
        if num_data_points==0:
            fnn = fn+'.pkl'
        else:
            fnn = fn+'_'+str(num_data_points)+'.pkl'
            
        with open(fnn, 'rb') as f:
            data = pi.load(f)
        
    return data
                
def run_net(FLAGS):
#     plot_data(MAX_EPISODES)
    root_dir = os.getcwd()
    summaries_dir = root_dir + '/Summaries';
    save_dir = root_dir;

    data_dict = import_data(False, fold=1, data_type='train', num_data_points=0)
    data_mgr = DataManager(data_dict)
    num_data_samples = len(data_dict)
    
    learning_rate=0.00001
    batch_size = 32
    dropout_val = 1.0
    use_simple_logreg = False
    use_softmax = True
    
    if use_simple_logreg:
        n_inputs = NUM_FEATURES * 2
        n_outputs = 1
        x, y = build_net(n_inputs, n_outputs)
        targets = tf.placeholder(tf.float32, [None, n_outputs], 'targets') # 0 or 1
        r_sum = tf.reduce_sum((-targets * tf.log(y+1e-10)) - ((1 - targets) * tf.log(1 - y + 1e-10)), 
                                                        reduction_indices=[1])
        loss = tf.reduce_mean(r_sum)
        
        # Test trained model
        half_vec = tf.fill(tf.shape(y), 0.5)
        gt_pred = tf.to_int32(tf.greater(y, half_vec, 'gt_pred'))
        gt_true = tf.to_int32(tf.greater(targets, half_vec, 'gt_true'))
        num_one_preds = tf.reduce_sum(gt_pred)
        fraction_ones = tf.reduce_mean(tf.to_float(gt_pred))
        fraction_ones_true = tf.reduce_mean(tf.to_float(gt_true))
    #     argm_y = tf.argmax(y, 1)
    #     argm_y_ = tf.argmax(y_, 1)
        correct_prediction = tf.equal(gt_pred, gt_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    elif use_softmax:
        n_inputs = NUM_FEATURES
        n_outputs = 5
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        x, y, loss, targets = build_net_softmax(n_inputs, n_outputs, [64], 
                                                keep_prob, lambda_val=0.0)

        # Test trained model
        argm_y = tf.to_int32( tf.argmax(y, 1) )
        max_y = tf.reduce_max(y, 1) 
        correct_prediction = tf.equal(argm_y, targets)
        accuracy_vec = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(accuracy_vec)


    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#     train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    
    save_model_interval = 100 #in epochs

    path_arr = [FLAGS.model, "lr{}".format(learning_rate)]

    with tf.Session() as sess:    
        
        if FLAGS.eval: #Restore saved model   
            fn= FLAGS.model + '_' + FLAGS.game
            model_file_name = root_dir + '/final_models/' + fn + '.ckpt'  
            print('loading model from: ' + model_file_name)  
            saver2restore = tf.train.Saver(write_version=1)
            saver2restore.restore(sess, model_file_name)
            return

        merged = tf.summary.merge_all()
        dir_name = summaries_dir +'/' + str(random.randint(0,99)) + '/lr_' + str(learning_rate)
#         train_writer = tf.summary.FileWriter(dir_name + '/train', sess.graph)

        tf.global_variables_initializer().run()    
        start = timer()

        for epoch in range(1000000):
#             config = Config(max_depth=3)
#             graphviz = GraphvizOutput(output_file='filter_exclude.png')
#             with PyCallGraph(output=graphviz, config=config):

            for i in range(num_data_samples // batch_size):

                input_vals, target_vals = data_mgr.sample(batch_size, n_inputs, use_simple_logreg=use_simple_logreg, use_softmax=use_softmax)
                
                accuracy_val, y_val, loss_val,_ = \
                    sess.run([accuracy, y, loss, train_step], 
                             feed_dict={x: input_vals, targets: np.squeeze(target_vals), keep_prob: dropout_val})
#             fraction_ones_true_val, fraction_ones_val, num_one_preds_val, accuracy_val, half_vec_val, \
#             gt_pred_val, gt_true_val, r_sum_val, y_val, loss_val,_ = \
#                 sess.run([fraction_ones_true,fraction_ones,num_one_preds,accuracy,half_vec, gt_pred, 
#                           gt_true, r_sum, y, loss, train_step], 
#                          feed_dict={x: input_vals, targets: target_vals})

            if epoch % 1 == 0:
                input_vals, target_vals = data_mgr.sample(1000, n_inputs, use_simple_logreg=use_simple_logreg, use_softmax=use_softmax)
                accuracy_val = sess.run(accuracy, feed_dict={x: input_vals, targets: np.squeeze(target_vals), keep_prob: 1.0})
                
#                 print('{} loss {} accuracy {} num_one_preds_val {} fraction_ones_val {}'.format(epoch, loss_val, accuracy_val, num_one_preds_val, fraction_ones_val))
                print('{} loss {} accuracy {}'.format(epoch, loss_val, accuracy_val))
#             res = summary_pb2.Summary.Value(tag="residual", simple_value=residual_val)
#             lo = summary_pb2.Summary.Value(tag="loss", simple_value=loss_val)
#             summary = summary_pb2.Summary(value=[lo, res])
#             train_writer.add_summary(summary, total_agent_steps)


#             if total_agent_steps % target_update_interval==0 and total_agent_steps>0: #update the target net
#                 print('{} updating target'.format(total_agent_steps))
#                 sess.run(update_target_op)

#                 end = timer()
#                 duration.append(end-start)
            
#             episode_length[episode] = t+1
#             rewards[episode] = cum_rewards
#             Qcheck_val = check_Q(sess, replay_buffer, Qfunc, Qfunc_tar, x, x_tar, residual)
# 
#             cr = summary_pb2.Summary.Value(tag="cum_rewards", simple_value=cum_rewards)
#             l = summary_pb2.Summary.Value(tag="length", simple_value=t+1)
#             qc = summary_pb2.Summary.Value(tag="Qcheck", simple_value=Qcheck_val)
#             summary = summary_pb2.Summary(value=[cr, l, qc])
#             train_writer.add_summary(summary, total_agent_steps)
# 
#             if episode % 10 == 0:
#                 end = timer()
#                 total_duration = end-start
#                 time_per_episode = total_duration / (episode+1)
#                 print('{} len {} cum_rewards {:.3f} Qc {:.2f} as {:g} t/ep {:.1f}'.format(episode, t+1, 
#                         cum_rewards, Qcheck_val, total_agent_steps, time_per_episode))
# 
#             if episode % save_model_interval == 0 and episode>0:
#                 #save trained model
#                 model_file_name = '_'.join(path_arr)+'_'+ str(episode) #write every episode
#                 save_model(sess, model_file_name, root_dir)
# 
#         print('length {:.6f} std: {:.6f}'.format(
#                                             np.asscalar(np.mean(episode_length)), 
#                                             np.asscalar(np.std(episode_length))))
#         print('cum reward {:.6f} std: {:.6f}'.format(
#                                             np.asscalar(np.mean(rewards)), 
#                                             np.asscalar(np.std(rewards))))

#         pi.dump( (episode_length, residuals, rewards), open( model_file_name+'.pi', "wb" ) )
        
#         (episode_length, losses, rewards) = pi.load( open( 'qA4_data', "rb" ) )
    
#         x_s = np.arange(0, MAX_EPISODES, 1)
#         
#         plt.figure(1, figsize=(12, 8))
#         plt.subplot(311)
#         plt.plot(x_s, np.mean(residuals, axis=0), 'r-')
#         plt.title("Double Q: " + 
#                   ' mean episode length: ' + str(np.asscalar(np.mean(episode_length))))
#         plt.ylabel('Absolute Residual')
#         
#         plt.subplot(312)
#         plt.plot(x_s, np.mean(episode_length, axis=0), 'b-')
#         plt.ylabel('Episode length')
#         
#         plt.subplot(313)
#         plt.plot(x_s, np.mean(rewards, axis=0), 'g-')
#         plt.xlabel('episode #')
#         plt.ylabel('Return')
#         plt.tight_layout()
#         plt.savefig('Fig_' + model_file_name)
#         plt.show() 


def plot_data(n_episodes):
    (episode_length_f, residuals_f, rewards_f) = pi.load( open( 'qA8_rep10_DQFalse_0.pi', "rb" ) )
    (episode_length_t, residuals_t, rewards_t) = pi.load( open( 'qA8_rep10_DQTrue_0.pi', "rb" ) )

    x_interval = 20    

    x_s = np.arange(0, n_episodes, x_interval)
    
    fig = plt.figure(1, figsize=(15, 5))
#     plt.title("No hidden units: "+str(n_hidden))
    plt.subplot(311)
    plt.plot(x_s, np.mean(residuals_t, axis=0)[0::x_interval], 'r-',label='Double Q learning')
    plt.plot(x_s, np.mean(residuals_f, axis=0)[0::x_interval], 'b--',label='no Double Q learning', linewidth=0.5)
    plt.ylabel('Absolute Residual')
    plt.legend(loc=0, borderaxespad=1.)
    
    print('residuals: {:.1f} {:.1f}'.format(np.asscalar(np.mean(residuals_t)),
                               np.asscalar(np.mean(residuals_f))))
    print('episode length: {:.1f} {:.1f}'.format(np.asscalar(np.mean(episode_length_t)),
                               np.asscalar(np.mean(episode_length_f))))
    print('returns: {:.3f} {:.3f}'.format(np.asscalar(np.mean(rewards_t)),
                               np.asscalar(np.mean(rewards_f))))
    plt.subplot(312)
    plt.plot(x_s, np.mean(episode_length_t, axis=0)[0::x_interval], 'r-',label='frozen target update')
    plt.plot(x_s, np.mean(episode_length_f, axis=0)[0::x_interval], 'b--',label='no frozen target', linewidth=0.5)
    plt.ylabel('Episode length')
    
    plt.subplot(313)
    plt.plot(x_s, np.mean(rewards_t, axis=0)[0::x_interval], 'r-',label='frozen target update')
    plt.plot(x_s, np.mean(rewards_f, axis=0)[0::x_interval], 'b--',label='no frozen target', linewidth=0.5)
    plt.xlabel('episode #')
    plt.ylabel('Return')
    
#     fig.legend((res20,res100,res1000), ('30 hidden units','100 hidden units','1000 hidden units'), 'lower right')
    plt.tight_layout()
    plt.show() 
    exit()
     
def clipped_reward(r_t1):
    if r_t1 == 0:
        cr = 0
    elif r_t1 < 0:
        cr = -1
    else:
        cr = 1
    return cr
        
def get_epsilon_greedy_action(sess, Qfunc, x, s, epsilon, n_actions):
    if random.random() < epsilon: #explore
        return random.randint(0,n_actions-1)
    else: #exploit
        s = np.expand_dims(s, axis=0)
        Qfunc_s_t = sess.run(Qfunc, feed_dict={x: s})
        max_a = np.argmax(Qfunc_s_t, axis=1)
        return max_a[0]
