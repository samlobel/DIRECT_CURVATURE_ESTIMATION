import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


BATCH_SIZE = 2048
LR_ORIG = 0.0001
LR_BY_ITSELF = 0.04
# LR_ORIG: CORRECT_D FOR FIRST ENTRY
# 0.01 : 0.0661
# 0.001 : 0.0046
# 0.0001 : 0.000444


# LR_ORIG : SECOND_DERIVATIVE
# 0.01 : -267.374267578
# 0.001 : -3855.65405273
# 0.0001 : -39862.2070312

# So the problem is CLEARLY in the second derivative calcualtion.

eps = 1e-4

print('assigning placeholders')
x = tf.placeholder(tf.float32, [BATCH_SIZE, 784])
y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

print('creating variables')
W1 = tf.Variable(tf.random_uniform([784, 10], -0.001,0.001), dtype=tf.float32)
b1 = tf.Variable(tf.random_uniform([10], -0.001,0.001), dtype=tf.float32)

y = tf.nn.softmax(tf.matmul(x,W1) + b1)


# VAR_LIST = [W1, b1]
print('creating loss and grads')

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.maximum(y,1e-4)))
grad_w1 = tf.gradients(cross_entropy, W1)[0]
grad_b1 = tf.gradients(cross_entropy, b1)[0]


print('making new values')
new_w1 = W1 - LR_ORIG*grad_w1
new_b1 = b1 - LR_ORIG*grad_b1

new_y = tf.nn.softmax(tf.matmul(x,new_w1) + new_b1)

new_ce = -tf.reduce_sum(y_*tf.log(tf.maximum(new_y,1e-4)))

DIFF = new_ce - cross_entropy

GRAD_SIZE_SQUARED = tf.reduce_sum(tf.square(grad_w1)) + tf.reduce_sum(tf.square(grad_b1))

second_derivative = 2*tf.abs((tf.abs(DIFF) - tf.abs(LR_ORIG*GRAD_SIZE_SQUARED)) / (LR_ORIG*LR_ORIG*GRAD_SIZE_SQUARED))

GRAD_SIZE = tf.pow(GRAD_SIZE_SQUARED, 0.5)

correct_d_unminned = tf.abs(tf.div(GRAD_SIZE, tf.abs(second_derivative)+eps)) / 100
correct_d = tf.minimum(0.04, correct_d_unminned)

# That's how far it could go before it erased it. Let's go half that. Although, if something's really flat,
# this could cause bigtime problems. Like learning rate of 1000. I need to investigate the values involved.

update_w1 = tf.assign(W1, W1  - correct_d*grad_w1)
update_b1 = tf.assign(b1, b1  - correct_d*grad_b1)

# update_w1 = tf.assign(W1, W1  - LR_BY_ITSELF*grad_w1)
# update_b1 = tf.assign(b1, b1  - LR_BY_ITSELF*grad_b1)

updates = [update_w1, update_b1]


if __name__ == '__main__':
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    _cross_entropy, _diff, _grad_size_squared, _grad_size, _second_derivative, _correct_d, _update_w1, _update_b1 = sess.run(
      [cross_entropy, DIFF, GRAD_SIZE_SQUARED, GRAD_SIZE, second_derivative, correct_d, update_w1, update_b1],
      feed_dict={x: batch_xs, y_: batch_ys}
    )
    if i % 10 == 0: 
      # print('DIFF: {}\nGRAD_SIZE_SQUARED: {}\nGRAD_SIZE: {}\nCORRECT_D: {}\nSECOND_DERIVATIVE: {}\n\n\n'.format(_diff, _grad_size_squared, _grad_size, _correct_d, _second_derivative))
      print('CORRECT_D: {:4.5f}'.format(_correct_d))
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
      print('ACC: {:4.5f}'.format(acc))

    if i == 2000:
      print(i)
      exit()
    # if (i+1) % 2000 == 0:
    #   print(i)
    #   exit()
    
    # _cross_entropy = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
    # _ = sess.run(update_not_scaling, feed_dict={x: batch_xs, y_: batch_ys})
    # _ = sess.run(update_scaling, feed_dict={x: batch_xs, y_: batch_ys})

    # _, _cross_entropy = sess.run([train_scaling, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    # _b2_grads, _mean, _stddev, _diff_grads = sess.run([b2_grads, mean, stddev, diff_grads], feed_dict={x: batch_xs, y_: batch_ys})
    # print(_b2_grads)
    # print("MEAN: {}      STDDEV:{}".format(_mean,_stddev))
    # print('diff grads should be around zero... {}'.format(_diff_grads))
    # exit()
    # sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # if i % 10 == 0:
    #   # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    #   acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    #   print("i: {}    CE: {}    ACC: {}".format(i, _cross_entropy, acc))
      
      # print("i: {}    ACC:   {}\n".format(acc))
      # print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))

























