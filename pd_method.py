import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util

train=True

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            weight=tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram(layer_name+'/weight',weight)
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.random_normal([1,out_size]))
            tf.summary.histogram(layer_name+'/biases',biases)
        wx_plus_b=tf.matmul(inputs,weight)+biases
        if activation_function is None:
            outputs=tf.add(tf.matmul(inputs,weight),biases,name='output')
        else:
            outputs=activation_function(wx_plus_b,name='output')
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x-input')
    ys=tf.placeholder(tf.float32,[None,1],name='y-input')

l1=add_layer(xs,1,10,1,activation_function=tf.nn.sigmoid)
prediction=add_layer(l1,10,1,2,None)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)

#优化算法
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init=tf.initialize_all_variables()

saver=tf.train.Saver()
with tf.Session() as sess:
    merged=tf.summary.merge_all()
    if train:
        writer=tf.summary.FileWriter('logs/',sess.graph)
        sess.run(init)
        for i in range(1000):
            sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
            if i%50==0:
                result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
                writer.add_summary(result,i)
                print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            if i==999:
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
                with tf.gfile.FastGFile('./pd/', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
    else:
        model_file=tf.train.latest_checkpoint('./ckpt/')
        saver.restore(sess,model_file)
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))


