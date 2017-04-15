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
from enum import Enum

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

def build_net(n_inputs, n_outputs, layer_sizes, keep_prob):
    
    x = tf.placeholder(tf.float32, [None, n_inputs], 'x')
#     previous_layer = x
    previous_layer = tf.nn.dropout(x, keep_prob)
    
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
    
    return x, y, logits

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
        
    def num_documents(self):
        count = 0
        for qid, rfps in self.dict.items():
            count += len(rfps)
        return len(self.dict), count
    
    def sample(self, batch_size, n_inputs, model_type):
        
        inputs = np.zeros((batch_size, n_inputs))
        targets = np.zeros((batch_size, 1))
        for i in range(batch_size):
            qid = random.sample(list(self.dict), 1)[0]
            rfps = self.dict[qid]
            rfp1 = random.sample(rfps, 1)[0]

            if model_type == ModelType.LOG_REG_NO_MATCH:
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
                    
            elif model_type == ModelType.LOG_REG_SUBTRACT:
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
                
                inputs[i,:] = np.subtract(rfp1_rnd.features,rfp2_rnd.features)
                if rfp1_rnd.relevance > rfp2_rnd.relevance:
                    targets[i,:] = 1
                else:
                    targets[i,:] = 0
                    
            elif model_type == ModelType.SOFTMAX_5:
                inputs[i,:] = rfp1.features
#                 targets[i,:] = np.zeros((NUM_RELEVANCE_LEVELS))
#                 targets[i,rfp1.relevance] = 1
                targets[i,0] = rfp1.relevance
                
            elif model_type == ModelType.SOFTMAX_3:
                rfp2 = random.sample(rfps, 1)[0]
                
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
                elif rfp1_rnd.relevance < rfp2_rnd.relevance:
                    targets[i,:] = 2
                else:
                    targets[i,:] = 0
                
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
    
def normalize(data, data_type, use_saved):
    n = len(data)
    n_inputs = len(data[0].features)
    features = np.zeros((n, n_inputs))
    
    for i, d in enumerate(data):
        features[i,:] = d.features

    if not use_saved and data_type == 'train':
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

        normalize(data, data_type, use_saved=True)
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
    else:
        if num_data_points==0:
            fnn = fn+'.pkl'
        else:
            fnn = fn+'_'+str(num_data_points)+'.pkl'
            
        with open(fnn, 'rb') as f:
            data = pi.load(f)
        
    return data
      
class ModelType(Enum):
    LOG_REG_NO_MATCH = 1
    SOFTMAX_5 = 2
    LOG_REG_SUBTRACT = 3 #3
    
    
def evaluate(num_qids, sess, keep_prob, x, y, argm_y, data_mgr, max_rank, model_type, ninputs):
    NDCGs = {}
    MAPs = {}
    MAPs_rnd = {}
    NDCGs_rnd = {}
    num_rnd_baselines = 100
#     num_qids = 1000000
    count=0
    
    for qid, rfps in data_mgr.dict.items():
        if count > num_qids:
            break
        count += 1
        
        #list of Relevance_Feature_Pairs
        inputs = []
        rels = []
        for rfp in rfps:
            inputs.append( rfp.features )
            rels.append(rfp.relevance)

        #need to perform ordering before calcing metrics
        if model_type == ModelType.LOG_REG_NO_MATCH or \
           model_type == ModelType.LOG_REG_SUBTRACT:
            rho = greedy_order(inputs, sess, keep_prob, x, y, model_type, ninputs)    
        elif model_type == ModelType.SOFTMAX_5:
            rho = sess.run(argm_y, feed_dict={x: inputs, keep_prob: 1.0})
        
        #now calc DCG and NDCG
        top_rels = []
        dcg = 0
        sorted_idxs = np.flipud( np.argsort(rho) )
        for i, idx in enumerate(sorted_idxs[:max_rank]): #now we're going down the rank of docs
            rel = rfps[idx].relevance
            dcg += ((2**rel) - 1)/np.log2(i + 2)
            top_rels.append(rel)
            
        #calc random baselines for MAP and NDCG
        dcg_rnd = np.zeros(num_rnd_baselines)
        for r in range(num_rnd_baselines):
            sorted_idxs = np.arange(len(rho))
            np.random.shuffle(sorted_idxs)
            for i, idx in enumerate(sorted_idxs[:max_rank]): #now we're going down the rank of docs
                rel = rfps[idx].relevance
                dcg_rnd[r] += ((2**rel) - 1)/np.log2(i + 2)
                    
            sum_Pk = 0
            num_relevant = 0
            relevances = []
            for k, idx in enumerate(sorted_idxs): #now we're going down the rank of docs
                rel = rfps[idx].relevance
                if rel > 0:
                    rel = 1 #convert to 1s and 0s
                    relevances.append(rel)
                    P_k = np.mean(relevances)
                    sum_Pk += P_k
                    num_relevant += 1
                else:
                    relevances.append(rel)
            
            AveP = sum_Pk / num_relevant
            MAPs_rnd[qid] = AveP
            
        #calc normalizer IDCG
        rels.sort(reverse=True)
        idcg=0
        for i, rel in enumerate(rels[:max_rank]):
            if rel < 1:
                break;
            idcg += ((2**rel) - 1)/np.log2(i + 2)
            
        NDCGs[qid] = dcg / idcg 
        NDCGs_rnd[qid] = np.mean(np.divide(dcg_rnd, idcg)) 
        
        #calc MAP
        sum_Pk = 0
        num_relevant = 0
        relevances = []
        sorted_idxs = np.flipud( np.argsort(rho) )
        for k, idx in enumerate(sorted_idxs): #now we're going down the rank of docs
            rel = rfps[idx].relevance
            if rel > 0:
                rel = 1 #convert to 1s and 0s
                relevances.append(rel)
                P_k = np.mean(relevances)
                sum_Pk += P_k
                num_relevant += 1
            else:
                relevances.append(rel)
        
        AveP = sum_Pk / num_relevant
        MAPs[qid] = AveP
        
    mean_NDCG = np.mean(list(NDCGs.values()))
    mean_NDCG_rnd = np.mean(list(NDCGs_rnd.values()))
    mean_MAP = np.mean(list(MAPs.values()))
    mean_MAP_rnd = np.mean(list(MAPs_rnd.values()))
        
#     print('mean NDCG {:.3f} +/- {:.2f}'.format(mean_NDCG, np.std(list(NDCGs.values()))))
        
    return mean_NDCG, mean_NDCG_rnd, mean_MAP, mean_MAP_rnd
    
def greedy_order(inputs, sess, keep_prob, x, y, model_type, ninputs):
    #implemented from Cohen et al Learning to Order Things
    n_samples = len(inputs)
    net_input = np.zeros((n_samples, ninputs))
    sum_prefs = np.zeros((n_samples, 2)) #pi(v) in paper
    pref_mtx = np.zeros((n_samples, n_samples)) #PREF(v,u) in paper
    rho = np.zeros((n_samples)) #rho(t) in paper
    
    for i in range(n_samples):
        sum_prefs[i,0] = i
        for j in range(n_samples): #batch_size x ninputs
#             net_input[j,:NUM_FEATURES] = inputs[i]
#             net_input[j,NUM_FEATURES:] = inputs[j]
            if model_type == ModelType.LOG_REG_NO_MATCH:
                net_input[j,:NUM_FEATURES] = inputs[i]
                net_input[j,NUM_FEATURES:] = inputs[j]
            elif model_type == ModelType.LOG_REG_SUBTRACT:
                net_input[j,:] = np.subtract(inputs[i],inputs[j])
            
            
        y_val = sess.run(y, feed_dict={x: net_input, keep_prob: 1.0})
        sum_prefs[i,1] += np.sum(y_val)
        pref_mtx[i,:] = np.squeeze(y_val)

        for j in range(n_samples): #batch_size x ninputs
            if model_type == ModelType.LOG_REG_NO_MATCH:
                net_input[j,:NUM_FEATURES] = inputs[j]
                net_input[j,NUM_FEATURES:] = inputs[i]
            elif model_type == ModelType.LOG_REG_SUBTRACT:
                net_input[j,:] = np.subtract(inputs[j],inputs[i])
            
        y_val = sess.run(y, feed_dict={x: net_input, keep_prob: 1.0})
        sum_prefs[i,1] -= np.sum(y_val)
        pref_mtx[:,i] = np.squeeze(y_val)
        
                
    #now do the ranking loop    
    while len(sum_prefs) > 0:
        idx = np.argmax(sum_prefs[:,1], axis=0)
        t = int(sum_prefs[idx,0])
        rho[t] = len(sum_prefs)
        sum_prefs = np.delete( sum_prefs, idx, axis=0)
    
        for i, v in enumerate(sum_prefs):
            vidx = int(v[0])
            v[1] += (pref_mtx[idx, vidx] - pref_mtx[vidx,idx])
    
    return rho
    
def run_net(FLAGS):
#     plot_data(MAX_EPISODES)
#     model_type = ModelType.SOFTMAX_5
#     model_type = ModelType.LOG_REG_NO_MATCH
    model_type = ModelType.LOG_REG_SUBTRACT
    root_dir = os.getcwd()
    summaries_dir = root_dir + '/Summaries';

    data_dict = import_data(True, fold=1, data_type='train', num_data_points=0)
    num_data_samples = len(data_dict)
    data_mgr_train = DataManager(data_dict)
    data_dict = import_data(True, fold=1, data_type='vali', num_data_points=0)
    data_mgr_vali = DataManager(data_dict)
#     data_dict = import_data(True, fold=1, data_type='test', num_data_points=0)
#     data_mgr_test = DataManager(data_dict)

#     q_train, doc_train = data_mgr_train.num_documents()
#     q_vali, doc_vali = data_mgr_vali.num_documents()
#     q_test, doc_test = data_mgr_test.num_documents()
    
    learning_rate=0.00001
    batch_size = 32
    dropout_val = 0.5
    hidden_units = [64,16]
    lambda_val = 0.0001
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    if model_type == ModelType.LOG_REG_NO_MATCH:
        n_inputs = NUM_FEATURES * 2
        n_outputs = 1
        x, y, logits = build_net(n_inputs, n_outputs, hidden_units, keep_prob)
        targets = tf.placeholder(tf.float32, [None,1], 'targets') # 0 or 1
        log_like = (-targets * tf.log(y+1e-10)) - ((1 - targets) * tf.log(1 - y + 1e-10))
        loss = tf.reduce_mean(log_like, axis=0)
        loss = tf.squeeze(loss)
#         regularizer = tf.nn.l2_loss(weights)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss = loss + lambda_val * l2_loss
        
        # Test trained model
        argm_y = None
        half_vec = tf.fill(tf.shape(y), 0.5)
        gt_pred = tf.to_int32(tf.greater(y, half_vec, 'gt_pred'))
        gt_true = tf.to_int32(tf.greater(targets, half_vec, 'gt_true'))
        correct_prediction = tf.equal(gt_pred, gt_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    if model_type == ModelType.LOG_REG_SUBTRACT:
        n_inputs = NUM_FEATURES
        n_outputs = 1
        x, y, logits = build_net(n_inputs, n_outputs, hidden_units, keep_prob)
        targets = tf.placeholder(tf.float32, [None,1], 'targets') # 0 or 1
        log_like = (-targets * tf.log(y+1e-10)) - ((1 - targets) * tf.log(1 - y + 1e-10))
        loss = tf.reduce_mean(log_like, axis=0)
        loss = tf.squeeze(loss)
#         regularizer = tf.nn.l2_loss(weights)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss = loss + lambda_val * l2_loss
        
        # Test trained model
        argm_y = None
        half_vec = tf.fill(tf.shape(y), 0.5)
        gt_pred = tf.to_int32(tf.greater(y, half_vec, 'gt_pred'))
        gt_true = tf.to_int32(tf.greater(targets, half_vec, 'gt_true'))
        correct_prediction = tf.equal(gt_pred, gt_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    elif model_type == ModelType.SOFTMAX_5:
        n_inputs = NUM_FEATURES
        n_outputs = 5
        x, y, logits = build_net(n_inputs, n_outputs, hidden_units, keep_prob)
        targets = tf.placeholder(tf.int32, [None], 'targets') # 0 or 1
        Xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
        loss = tf.reduce_mean(Xent_loss)
#         regularizer = tf.nn.l2_loss(weights)
#         l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
#         total_loss = loss + lambda_val * l2_loss

        # Test trained model
        argm_y = tf.to_int32( tf.argmax(y, 1) )
        correct_prediction = tf.equal(argm_y, targets)
        accuracy_vec = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(accuracy_vec)


    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#     train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    

    path_arr = ["lr{}".format(learning_rate),"lam{}".format(lambda_val),"drop{:.1f}".format(dropout_val), "nhid{}".format(hidden_units), str(model_type.name)]

    with tf.Session() as sess:    
        
        if FLAGS.eval: #Restore saved model   
            fn= FLAGS.model + '_' + FLAGS.game
            model_file_name = root_dir + '/final_models/' + fn + '.ckpt'  
            print('loading model from: ' + model_file_name)  
            saver2restore = tf.train.Saver(write_version=1)
            saver2restore.restore(sess, model_file_name)
            return

        dir_name = summaries_dir +'/' + str(random.randint(0,99)) + '/' + '_'.join(path_arr)
        train_writer = tf.summary.FileWriter(dir_name + '_vali', sess.graph)

        tf.global_variables_initializer().run()    
#         start = timer()

        for epoch in range(1000000):
#             config = Config(max_depth=3)
#             graphviz = GraphvizOutput(output_file='filter_exclude.png')
#             with PyCallGraph(output=graphviz, config=config):

#             for _ in range(num_data_samples // batch_size):
            for _ in range(1):

                input_vals, target_vals = data_mgr_train.sample(batch_size, n_inputs, model_type)
                
                if model_type == ModelType.LOG_REG_NO_MATCH:
                    accuracy_train_val, gt_pred_val, gt_true_val, loss_val,_ = \
                        sess.run([accuracy, gt_pred, 
                                  gt_true, loss, train_step], 
                                 feed_dict={x: input_vals, targets: target_vals, keep_prob: dropout_val})
                        
                elif model_type == ModelType.LOG_REG_SUBTRACT:
                    accuracy_train_val, gt_pred_val, gt_true_val, loss_val,_ = \
                        sess.run([accuracy, gt_pred, 
                                  gt_true, loss, train_step], 
                                 feed_dict={x: input_vals, targets: target_vals, keep_prob: dropout_val})
                        
                elif model_type == ModelType.SOFTMAX_5:
                    accuracy_train_val, loss_val,_ = \
                        sess.run([accuracy, loss, train_step], 
                                 feed_dict={x: input_vals, targets: np.squeeze(target_vals), keep_prob: dropout_val})
                

            if epoch % 500 == 0:
                input_vals, target_vals = data_mgr_vali.sample(10000, n_inputs, model_type)
                accuracy_vali_val, loss_vali_val = sess.run([accuracy,loss], feed_dict={x: input_vals, targets: target_vals, keep_prob: 1.0})
                if epoch % 1000 == 0:
                    NDCG, NDCG_rnd, MAP, MAP_rnd = evaluate(1000, sess, keep_prob, x, y, argm_y, data_mgr_vali, 10, model_type, n_inputs)
                
                lov = summary_pb2.Summary.Value(tag="loss_vali", simple_value=np.asscalar(np.mean(loss_vali_val)))
                lot = summary_pb2.Summary.Value(tag="loss_train", simple_value=np.asscalar(np.mean(loss_val)))
                ndcg = summary_pb2.Summary.Value(tag="NDCG", simple_value=NDCG)
                ndcg_rnd = summary_pb2.Summary.Value(tag="NDCG_rnd", simple_value=NDCG_rnd)
                mavp = summary_pb2.Summary.Value(tag="MAP", simple_value=MAP)
                mavp_rnd = summary_pb2.Summary.Value(tag="MAP_rnd", simple_value=MAP_rnd)
                accv = summary_pb2.Summary.Value(tag="accuracy_vali", simple_value=np.asscalar(np.mean(accuracy_vali_val)))
                acct = summary_pb2.Summary.Value(tag="accuracy_train", simple_value=np.asscalar(np.mean(accuracy_train_val)))
                summary = summary_pb2.Summary(value=[lov,lot,ndcg,ndcg_rnd,accv,acct,mavp,mavp_rnd])
                train_writer.add_summary(summary, epoch)
                if epoch % 50 == 0:
                    print('{} loss {:.3f} accuracy {:.3f} NDCG {:.3f}'.format(epoch, loss_val, accuracy_vali_val, NDCG))

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
