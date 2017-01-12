import tensorflow as tf
import numpy as np
from models import one_layer_model, two_layer_model
# from models import two_layer_model

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


BATCH_SIZE = 2048
# LR_ORIG = 0.000000001
LR_ORIG = 1e-6
MAX_LR = 0.1

model = one_layer_model(LR_ORIG, MAX_LR)
# model = two_layer_model(LR_ORIG, MAX_LR)




init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)  

for i in xrange(10000):
  batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
  feed_dict = {
    model['input'] : batch_xs,
    model['target_output'] : batch_ys
  }
  sess.run(model['updates'], feed_dict=feed_dict)
  if i % 10 == 0:
    feed_dict = {
      model['input'] : mnist.test.images,
      model['target_output'] : mnist.test.labels
    }
    _cross_entropy, _accuracy, _second_derivative, _second_derivative_sigmoided, _grad_size_squared  = sess.run([ model['cross_entropy'],
                                              model['accuracy'],
                                              model['second_derivative'],
                                              model['second_derivative_sigmoided'],
                                              model['grad_size_squared'],
                                           ], feed_dict=feed_dict)
    # print('CE: {}\t\tACC: {}'.format(_cross_entropy, _accuracy))
    print('i: {}\tCE: {}\tACC: {}\t2ND_DER: {}\tGRAD_SIZE_SQUARED: {}'.format(i, _cross_entropy, _accuracy, _second_derivative, _grad_size_squared))


