import numpy as np
import tensorflow as tf

import keras.backend as K

from CC_ResNet.convolutional_block import convolutional_block
from CC_ResNet.identity_block import identity_block


# identity_block
tf.reset_default_graph()
with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
cond = np.allclose(out[0][1][1][0], [0.19716819, 0., 1.3561226, 2.1713073, 0., 1.3324987])
if cond:
    print("Test identity_block is OK")
else:
    print("Test identity_block FAILS")


#convolutional_block
tf.reset_default_graph()
with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = convolutional_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
cond = np.allclose(out[0][1][1][0], [0.09018463, 1.2348979, 0.46822023, 0.03671762, 0., 0.65516603])
if cond:
    print("Test convolutional_block is OK")
else:
    print("Test convolutional_block FAILS")