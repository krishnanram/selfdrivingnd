
import tensorflow as tf


def sample1() :
    x = tf.placeholder(tf.string)
    y = tf.placeholder(tf.int32)
    z = tf.placeholder(tf.float32)

    sess = tf.Session()

    with ( sess ) :
        print (sess.run(y, feed_dict={y : 132 } ) )

    x = tf.constant(10)
    y = tf.constant(2)
    z = tf.sub(tf.div(x,y),tf.constant(1))

    # TODO: Print z from a session
    with tf.Session() as sess:
        output = sess.run(z)
        print(output)


def classify1() :

    print ("asdsa")

def tsoftmax():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)

    softmax = tf.nn.softmax(logits)

    with tf.Session() as sess:
        output = sess.run(softmax, feed_dict={logits: logit_data})

    return output


import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


logits = [3.0, 1.0, 0.2]
print(softmax(logits))

print (tsoftmax())

a = 1000000000
for i in range(1000000):
    a = a + 1e-6
print(a - 1000000000)

