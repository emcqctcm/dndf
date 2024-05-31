import numpy as np
import tensorflow as tf
import scipy.io
import random
import tensorflow.contrib.slim as slim
import keras
import pandas as pd
import umap
from umap import UMAP
from keras.utils import to_categorical
#from capsule20200913 import Capsule


def init_w(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())


def init_b(shape, name):
    return tf.get_variable(name, shape, initializer=tf.zeros_initializer())


def fully_connected_linear(net, shape, appendix, pruner=None):
    with tf.name_scope("Linear{}".format(appendix)):
        w_name = "W{}".format(appendix)
        w = init_w(shape, w_name)
        if pruner is not None:
            w = pruner.prune_w(*pruner.get_w_info(w))
        b = init_b(shape[1], "b{}".format(appendix))
        return tf.add(tf.matmul(net, w), b, name="Linear{}".format(appendix))


class DNDF:
    def __init__(self, n_class, n_tree=3, tree_depth=4):  #n_tree=16
        print('dbdf的init')
        self.n_class = n_class
        self.n_tree, self.tree_depth = n_tree, tree_depth
        self.n_leaf = 2 ** (tree_depth + 1)
        self.n_internals = self.n_leaf - 1

    def __call__(self, net, n_batch_placeholder, dtype="output", pruner=None):
        name = "DNDF_{}".format(dtype)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            flat_probabilities = self.build_tree_projection(dtype, net, pruner)
            routes = self.build_routes(flat_probabilities, n_batch_placeholder)
            features = tf.concat(routes, 1, name="Feature_Concat")
            #if dtype == "feature":
            #    return features
            leafs = self.build_leafs()
            leafs_matrix = tf.concat(leafs, 0, name="Prob_Concat")
            return tf.divide(
                tf.matmul(features, leafs_matrix),
                float(self.n_tree), name=name
            )

    def build_tree_projection(self, dtype, net, pruner):
        with tf.name_scope("Tree_Projection"):
            flat_probabilities = []
            fc_shape = net.shape[1].value
            print(fc_shape)
            for i in range(self.n_tree):
                with tf.name_scope("Decisions"):
                     
                    p_left = tf.nn.sigmoid(fully_connected_linear(
                        net=net,
                        shape=[fc_shape, self.n_internals],
                        appendix="_tree_mapping{}_{}".format(i, dtype), pruner=None#pruner  #
                    ))
                    
                    #p_left=net[i]  #不经过全连接层
                    p_right = 1 - p_left
                    p_all = tf.concat([p_left, p_right], 1)
                    flat_probabilities.append(tf.reshape(p_all, [-1]))
        return flat_probabilities

    def build_routes(self, flat_probabilities, n_batch_placeholder):
        with tf.name_scope("Routes"):
            n_flat_prob = 2 * self.n_internals
            batch_indices = tf.reshape(
                tf.range(0, n_flat_prob * n_batch_placeholder, n_flat_prob),
                [-1, 1]
            )
            n_repeat, n_local_internals = int(self.n_leaf * 0.5), 1
            increment_mask = np.repeat([0, self.n_internals], n_repeat)
            routes = [
                tf.gather(p_flat, batch_indices + increment_mask)
                for p_flat in flat_probabilities
            ]
            for depth in range(1, self.tree_depth + 1):
                n_repeat = int(n_repeat * 0.5)
                n_local_internals *= 2
                increment_mask = np.repeat(np.arange(
                    n_local_internals - 1, 2 * n_local_internals - 1
                ), 2)
                increment_mask += np.tile([0, self.n_internals], n_local_internals)
                increment_mask = np.repeat(increment_mask, n_repeat)
                for i, p_flat in enumerate(flat_probabilities):
                    routes[i] *= tf.gather(p_flat, batch_indices + increment_mask)
        return routes

    def build_leafs(self):
        with tf.name_scope("Leafs"):
            if self.n_class == 1:
                local_leafs = [
                    init_w([self.n_leaf, 1], "RegLeaf{}".format(i))
                    for i in range(self.n_tree)
                ]
            else:
                local_leafs = [
                    tf.nn.softmax(w, name="ClfLeafs{}".format(i))
                    for i, w in enumerate([
                        init_w([self.n_leaf, self.n_class], "RawClfLeafs")
                        for _ in range(self.n_tree)
                    ])
                ]
        return local_leafs
    

    



class Pruner:
    def __init__(self, alpha=None, beta=None, gamma=None, r=1., eps=1e-12, prune_method="soft_prune"):
        self.alpha, self.beta, self.gamma, self.r, self.eps = alpha, beta, gamma, r, eps
        self.org_ws, self.masks, self.cursor = [], [], -1
        self.method = prune_method
        if prune_method == "soft_prune" or prune_method == "hard_prune":
            if alpha is None:
                self.alpha = 1e-2
            if beta is None:
                self.beta = 1
            if gamma is None:
                self.gamma = 1
            if prune_method == "hard_prune":
                self.alpha *= 0.01
            self.cond_placeholder = None
        elif prune_method == "surgery":
            if alpha is None:
                self.alpha = 1
            if beta is None:
                self.beta = 1
            if gamma is None:
                self.gamma = 0.0001
            self.r = None
            self.cond_placeholder = tf.placeholder(tf.bool, (), name="Prune_flag")
        else:
            raise NotImplementedError("prune_method '{}' is not defined".format(prune_method))

    @property
    def params(self):
        return {
            "eps": self.eps, "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
            "max_ratio": self.r, "method": self.method
        }

    @staticmethod
    def get_w_info(w):
        with tf.name_scope("get_w_info"):
            w_abs = tf.abs(w)
            w_abs_mean, w_abs_var = tf.nn.moments(w_abs, None)
            return w, w_abs, w_abs_mean, tf.sqrt(w_abs_var)

    def prune_w(self, w, w_abs, w_abs_mean, w_abs_std):
        self.cursor += 1
        self.org_ws.append(w)
        with tf.name_scope("Prune"):
            if self.cond_placeholder is None:
                log_w = tf.log(tf.maximum(self.eps, w_abs / (w_abs_mean * self.gamma)))
                if self.r > 0:
                    log_w = tf.minimum(self.r, self.beta * log_w)
                self.masks.append(tf.maximum(self.alpha / self.beta * log_w, log_w))
                return w * self.masks[self.cursor]

            self.masks.append(tf.Variable(tf.ones_like(w), trainable=False))

            def prune(i, do_prune):
                def sub():
                    if do_prune:
                        mask = self.masks[i]
                        self.masks[i] = tf.assign(mask, tf.where(
                            tf.logical_and(
                                tf.equal(mask, 1),
                                tf.less_equal(w_abs, 0.9 * tf.maximum(w_abs_mean + self.beta * w_abs_std, self.eps))
                            ),
                            tf.zeros_like(mask), mask
                        ))
                        mask = self.masks[i]
                        self.masks[i] = tf.assign(mask, tf.where(
                            tf.logical_and(
                                tf.equal(mask, 0),
                                tf.greater(w_abs, 1.1 * tf.maximum(w_abs_mean + self.beta * w_abs_std, self.eps))
                            ),
                            tf.ones_like(mask), mask
                        ))
                    return w * self.masks[i]
                return sub

            return tf.cond(self.cond_placeholder, prune(self.cursor, True), prune(self.cursor, False))

def attention(inputs, attention_size, time_major=False, return_alphas=False):
        
        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
    
        initializer = tf.random_normal_initializer(stddev=0.1)
    
        # Trainable parameters
        w_omega = tf.get_variable(name="w_omega", shape=[hidden_size, attention_size], initializer=initializer)
        b_omega = tf.get_variable(name="b_omega", shape=[attention_size], initializer=initializer)
        u_omega = tf.get_variable(name="u_omega", shape=[attention_size], initializer=initializer) #胶囊耦合向量
    
        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size，50   D=hidden size,150 T=SEQUENCE_LENGTH,250
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)  #v:(?,250,50)
    
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
    
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    
        if not return_alphas:
            return output
        else:
            return output, alphas


def weight_variable(shape):  
        # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0  
    initial = tf.truncated_normal(shape, stddev=0.1)  
    return tf.Variable(initial)  
def bias_variable(shape):  
        # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1  
    initial = tf.constant(0.1, shape=shape)  
    return tf.Variable(initial)  
def conv2d1(x, W):    
        # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘  
    return tf.nn.conv2d(x, W, strides=[1, 1, 6, 1], padding='VALID')


def conv2d2(x, W):    
        # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘  
    return tf.nn.conv2d(x, W, strides=[1, 1, 4, 1], padding='VALID')

def conv2d3(x, W):    
        # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘  
    return tf.nn.conv2d(x, W, strides=[1, 1, 16, 1], padding='VALID')
def conv2d4(x, W):    
        # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘  
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


components=16
batch_size=12  #12   #20
classnum=4  #2 
xarr=22*1125  #352
n_iterations_per_epoch =288#  v*8 #720*9  mnist.train.num_examples // batch_size  #修改前的值为20
n_iterations_validation =288 #814  mnist.validation.num_examples // batch_size   #修改前的值为20


'''后续添加的,以另一种方式实现第一、二个卷积层。意外，只是实现方法不一样，参数未变，准确率从5点提高到78点'''

xs = tf.placeholder(tf.float32, [batch_size,  xarr])   # 声明一个占位符，None表示输入图片的数量不定，6*250矩阵(28*28图片分辨率)
ys = tf.placeholder(tf.float32, [batch_size,classnum])  # 类别是0-1总共2个类别，对应输出分类结果   
keep_prob = tf.placeholder(tf.float32)  
#x_image = tf.reshape(xs, [batch_size,  10]) # x_image又把xs reshape成了28*28*1的形状，因为是灰色图片，所以通道是1.作为训练时的input，-1代表图片数量不定 
xs = tf.reshape(xs, [batch_size, 1, xarr])    

inputdata = scipy.io.loadmat('A1.mat')
inputdata_t = scipy.io.loadmat('A1.mat')

#数据集2
from preprocess5 import get_data

n_sub=1 #9
data_path =  "BCIiv2a/"
isStandard = True
LOSO = False
X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
    data_path, n_sub, LOSO, isStandard)  #[288,1,22,1125]


label_names = None

x=np.transpose(np.array(inputdata['x'][:,:,0:72]))  #12:14,GMM clustering on raw data,90%
x_t=np.transpose(np.array(inputdata_t['x'][:,:,72:144]))  #(72, 750, 22)

#y=np.array(inputdata['y'][0:72])
#y_t=np.array(inputdata_t['y'][72:144])
#train_labels  = y
#test_labels =y_t
#train_labels = keras.utils.to_categorical(train_labels, num_classes=2)
#test_labels = keras.utils.to_categorical(test_labels, num_classes=2)
train_labels =y_train_onehot
test_labels = y_test_onehot

def umaptrans(x):
    
    md = float(0.00)
    i=0
    x6=[]
    #x2=[x1]*2
    for i in range(0,22):
        x1=x[:,:,i]
        x2=x1.reshape(72,-1)
        x3 = umap.UMAP(
                random_state=0,
                metric='chebyshev',
                n_components=components,
                #n_epochs=11,
                n_neighbors=10,
                min_dist=md,
                #local_connectivity=1,
                ).fit_transform(x2, y)#(x2, y)
        
        
        x6.append(x3)
    x=(np.array(x6)).reshape(72,-1)
    return x

#train_data= umaptrans(x)   #(72,352)
#test_data= umaptrans(x_t)
train_data=X_train
test_data=X_test

# =============================================================================20220102
#W_conv1 = weight_variable([1, 5, 1, 32])    #卷积核1*8,
#b_conv1 = bias_variable([32]) # 对于每一个卷积核都有一个对应的偏置量。 
#conv1 = tf.nn.relu(conv2d1(X, W_conv1) + b_conv1)  #relu
# #conv1 =tf.nn.max_pool(conv1,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME')  
# 
#caps1_num  = 32
#caps1_dims = 8
# 
#W_conv2 = weight_variable([1, 1,32,64])   
#b_conv2 = bias_variable([64]) # 对于每一个卷积核都有一个对应的偏置量。 
#conv2= tf.nn.relu(conv2d2(conv1, W_conv2) + b_conv2)    #20*16*;relu


learning_rate = 0.01
n_conv_1 = 256 # 第一层16个ch
n_conv_2 = 128 # 第二层32个ch

X = tf.placeholder(shape=[batch_size, 1,xarr,1], dtype=tf.float32, name="X")

#conv2_flat = tf.reshape(conv2, [batch_size,64*(components-1) ])
conv2_flat = tf.reshape(X, [batch_size,xarr ])

#使用DNDF

W_fc1 = weight_variable([xarr, 64])  #普通全连接
b_fc1 = bias_variable([64])
h_fc1 = tf.nn.relu(tf.matmul(conv2_flat, W_fc1) + b_fc1)  #X
#W_fc1 = weight_variable([128, 64])  #普通全连接
#b_fc1 = bias_variable([64])
#h_fc1 = tf.nn.relu(tf.matmul(h_fc1, W_fc1) + b_fc1)  #X
# =============================================================================
h_fc1_drop = tf.nn.dropout(h_fc1, 0.8)

#dndf = DNDF(classnum)  #建立对象_dndf,DNDF(self.n_class, **dndf_params)
#prediction=dndf(
#        h_fc1_drop, batch_size,  #at_drop5   self._wide_input, self._n_batch_placeholder
#        pruner=None  #这里可以暂时置为None，Pruner(),=self._dndf_pruner
#        )
#print('prediction',prediction)

# =============================================================================
# #全连接层输出
# 
W_fc2 = weight_variable([64, classnum])
b_fc2 = bias_variable([classnum])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

loss = tf.reduce_mean(tf.square(ys[:,0:classnum]-prediction[:,0:classnum]))
#loss= tf.reduce_mean(tf.reduce_sum(ys[:,0:classnum]*tf.log(prediction[:,0:classnum]),reduction_indices=[0]))


#使用梯度下降法

#optimizer = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False)
optimizer =tf.compat.v1.train.AdamOptimizer()
#training_op = optimizer.minimize(loss, name="training_op")


training_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct= tf.equal(tf.argmax(ys,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))



'''Training'''
n_epochs =60#60    #修改前的值为3000

best_loss_val = np.infty
 
def arrange_data(data, labels):
    output_data = list()
    output_labels = list()
    for idx in range(len(data)): #len(data):9
        for segment in data[idx]:
            output_data.append(np.expand_dims(segment, axis=2))
            if labels[idx][0] == 1:
                output_labels.append(0)
            else:
                output_labels.append(1)
    output_data = np.array(output_data)
    output_labels = np.array(output_labels)
    return output_data, output_labels      
      
def trial_evaluate(model, data, labels):
    acc = 0.0
    for idx in range(len(data)):
        test_data, test_label = arrange_data(np.expand_dims(data[idx], axis=0), np.expand_dims(labels[idx], axis=0))
        test_label = keras.utils.to_categorical(test_label, num_classes=2)
        loss, accuracy = model.evaluate(test_data, test_label)
        if accuracy > 0.5:
            acc += 1.0
    acc = acc/len(data)
    return acc            

savedir = "d:/python/log/"
saver   = tf.train.Saver(max_to_keep=1)





with tf.Session() as sess:       #stacked_auto_encoder.ckpt-29
    sess.run(init)               #stacked_auto_encoder.ckpt-29

    for epoch in range(n_epochs):
        for iteration in range(batch_size, n_iterations_per_epoch + 1):
            batch=([],[])
            batch0=[]
            b_tmp1=[]
            p = random.sample(range(0,n_iterations_per_epoch), batch_size) #399      ;(range(0,720*9), 20)
            for k in p:
                
                batch[0].append(train_data[k,:]) #train_x['train_x'][k],gabor[:,k]   (batch[0]).shape()
                b_tmp1=train_labels[k,:].tolist()
                batch[1].append(b_tmp1)#b_tmp1[0]+2*b_tmp1[1]
                #batch[1].append(train_y['train_y'][k])
            
            _, loss_train = sess.run(
                [training_op, loss],  #tf.reshape(batch[0],[20, 6, 250,1])
                feed_dict={X:(np.array(batch[0])).reshape([batch_size,1,xarr,1]), #.reshape([20,6,250, 1])  X: X_batch.reshape([-1, 28, 28, 1])
                           ys:batch[1]}) #y: y_batch,  mask_with_labels: True
            #无监督卷积自编码器训练[optimizer3, cost3];变分胶囊[optimizer4, cost4]

            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  end="")

        loss_vals = []
        acc_vals = []
        for iteration in range(batch_size, n_iterations_validation + 1):
            #X_batch, y_batch = mnist.validation.next_batch(batch_size)
            batch=([],[])
            b_tmp1=[]
            p = random.sample(range(0,n_iterations_validation), batch_size)   #(range(0,360), 20)
            for k in p:
                batch[0].append( test_data[k,:].tolist())
                b_tmp1=test_labels[k,:].tolist()
                batch[1].append(b_tmp1) #b_tmp1[0]+2*b_tmp1[1]
                #batch[1].append( train_y['train_y'][k])
            loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={X: (np.array(batch[0])).reshape([batch_size,1,xarr,1]),  #[20, 6,250, 1]{X: X_batch.reshape([-1, 28, 28, 1]),
                               ys: batch[1]})  #y_batch
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)           
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)      

        #print('epoch,loss_val,acc_val',epoch+1,loss_val,acc_val)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
                epoch + 1, acc_val * 100, loss_val," (improved)" if loss_val < best_loss_val else ""))
        # And save the model if it improved:
        if loss_val < best_loss_val:
            #save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val
              
    
