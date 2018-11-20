import numpy as np
import tensorflow as tf
a=np.array([[1,2,3],[5,6,7]],dtype=np.float32)
b=tf.Variable(a)
d=tf.scatter_update(b,[0,0],[[3,3,3],[3,3,3]])
c=tf.norm(b,axis=1,ord=1)




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(d)
    print(b.eval())





