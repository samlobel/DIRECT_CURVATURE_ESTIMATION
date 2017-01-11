import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


BATCH_SIZE=32
LR = 0.01

def global_softmax(mat, temp=-1.0):
  exp_mat = tf.exp(temp*mat)
  normed = tf.div(exp_mat, tf.reduce_sum(exp_mat))
  return normed

def get_vec_len(mat):
  return tf.reduce_sum(tf.mul(mat,mat))

def get_grad_stats(unpacked_error, var):
  grads_unpacked = [tf.gradients(err, var)[0] for err in unpacked_error]
  # zipped = zip(grads_unpacked) #This should be list of lists.
  grads_packed = tf.pack(grads_unpacked)
  mean, stddev = tf.nn.moments(grads_packed, [0])
  return mean, stddev

def get_scaled_grad(err, unpacked_err, var):
  real_grad = tf.gradients(err, var)[0]
  # print(real_grad)
  # return real_grad
  mean, stddev_maybe_zero = get_grad_stats(unpacked_err, var)
  print(stddev_maybe_zero)
  to_mul = global_softmax(stddev_maybe_zero, -5.0)
  normed = tf.div(to_mul, get_vec_len(to_mul))
  stddev = tf.mul(normed, get_vec_len(stddev_maybe_zero))
  scaled_grad = real_grad * stddev
  return scaled_grad

def apply_regular_grad(err, var, LR):
  real_grad = tf.gradients(err, var)[0]
  update = tf.assign(var, var - LR*real_grad)
  return update

def apply_scaled_grad(err, unpacked_err, var, LR):
  scaled_grad = get_scaled_grad(err, unpacked_err, var)
  update = tf.assign(var, var - LR*scaled_grad)
  return update



x = tf.placeholder(tf.float32, [BATCH_SIZE, 784])
y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 10])


W1 = tf.Variable(tf.random_uniform([784, 200], -0.001,0.001))
b1 = tf.Variable(tf.random_uniform([200], -0.001,0.001))
h1 = tf.nn.elu(tf.matmul(x,W1) + b1)

W2 = tf.Variable(tf.random_uniform([200, 10], -0.001,0.001))
b2 = tf.Variable(tf.random_uniform([10], -0.001,0.001))
# h2 = tf.nn.elu(tf.matmul(h1,W2) + b2)
# y_bad = tf.nn.softmax(tf.matmul(h1,W2) + b2)
# y = tf.maximum(y_bad, 1e-6)
y = tf.nn.softmax(tf.matmul(h1,W2) + b2)


# W3 = tf.Variable(tf.random_uniform([50, 10], -0.001,0.001))
# b3 = tf.Variable(tf.random_uniform([10], -0.001,0.001))
# y_bad = tf.nn.softmax(tf.matmul(h2,W3) + b3)
# y = tf.maximum(y_bad, 1e-6)




# y = tf.nn.softmax(tf.matmul(x,W1))

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
ce_not_reduced = -tf.reduce_sum(y_*tf.log(y), reduction_indices=[1])
ce_unpacked = tf.unpack(ce_not_reduced)

# VAR_LIST = [W1,b1,W2,b2,W3,b3]
VAR_LIST = [W1,b1,W2,b2]

update_scaling = [apply_scaled_grad(cross_entropy, ce_unpacked, v, LR) for v in VAR_LIST]
update_not_scaling = [apply_regular_grad(cross_entropy, v, LR) for v in VAR_LIST]

# train_scaling = apply_scaled_grad(cross_entropy, ce_unpacked, W1, 0.01)
# train_regular = apply_regular_grad(cross_entropy, W1, 0.01)


if __name__ == '__main__':
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    _cross_entropy = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
    _ = sess.run(update_not_scaling, feed_dict={x: batch_xs, y_: batch_ys})
    # _ = sess.run(update_scaling, feed_dict={x: batch_xs, y_: batch_ys})

    # _, _cross_entropy = sess.run([train_scaling, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    # _b2_grads, _mean, _stddev, _diff_grads = sess.run([b2_grads, mean, stddev, diff_grads], feed_dict={x: batch_xs, y_: batch_ys})
    # print(_b2_grads)
    # print("MEAN: {}      STDDEV:{}".format(_mean,_stddev))
    # print('diff grads should be around zero... {}'.format(_diff_grads))
    # exit()
    # sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    if i % 10 == 0:
      # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
      acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
      print("i: {}    CE: {}    ACC: {}".format(i, _cross_entropy, acc))
      
      # print("i: {}    ACC:   {}\n".format(acc))
      # print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))


















# import tensorflow as tf
# import numpy as np

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# BATCH_SIZE=25


# def get_grad_stats(unpacked_error, var_list):
#   grads_unpacked = [tf.gradients(err, var_list) for err in unpacked_error]
#   zipped = zip(grads_unpacked) #This should be list of lists.
#   grads_packed = tf.pack(grads_unpacked)
#   mean, stddev = tf.nn.moments(grads_packed, [0])
#   return mean, stddev

# def get_scaled_grad(err, unpacked_err, var_list):
#   real_grad = tf.gradients(err, var_list)
#   mean, stddev = get_grad_stats(unpacked_err, var_list)
#   stddev_with_mean_one = stddev / tf.reduce_mean(stddev)
#   scaled_grad = real_grad * stddev_with_mean_one






# x = tf.placeholder(tf.float32, [BATCH_SIZE, 784])
# y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

# W1 = tf.Variable(tf.random_uniform([784, 200], -0.001,0.001))
# b1 = tf.Variable(tf.random_uniform([200], -0.001,0.001))
# h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

# W2 = tf.Variable(tf.random_uniform([200, 10], -0.001,0.001))
# b2 = tf.Variable(tf.random_uniform([10], -0.001,0.001))
# y = tf.nn.softmax(tf.matmul(h1,W2) + b2)
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# ce_not_reduced = -tf.reduce_sum(y_*tf.log(y), reduction_indices=[1])
# unpacked_ce = tf.unpack(cross_entropy_not_reduced)

# print('y_ shape: ')
# print(tf.shape(y_))
# W1 = tf.Variable(tf.random_uniform([784, 200], -0.001,0.001))
# b1 = tf.Variable(tf.random_uniform([200], -0.001,0.001))
# h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

# W2 = tf.Variable(tf.random_uniform([200, 10], -0.001,0.001))
# b2 = tf.Variable(tf.random_uniform([10], -0.001,0.001))
# y = tf.nn.softmax(tf.matmul(h1,W2) + b2)
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# train_step = tf.train.MomentumOptimizer(0.01, 0.1).minimize(cross_entropy)

# cross_entropy_not_reduced = -tf.reduce_sum(y_*tf.log(y), reduction_indices=[1])
# unpacked_ce = tf.unpack(cross_entropy_not_reduced)
# print(cross_entropy_not_reduced.get_shape())
# # b2_grads = tf.gradients(unpacked_ce, b2)
# print('about to make b2_grads_unpacked')
# b2_grads_unpacked = [tf.gradients(ce, b2) for ce in unpacked_ce]
# print('made. about to repack')
# b2_grads = tf.pack(b2_grads_unpacked)
# mean, stddev = tf.nn.moments(b2_grads, [0])
# print('unpacked')
# av_grad = tf.reduce_sum(b2_grads, reduction_indices = [0])
# av_grad_also = tf.gradients(cross_entropy, b2)
# diff_grads = av_grad - av_grad_also

# if __name__ == '__main__':
#   init = tf.initialize_all_variables()
#   sess = tf.Session()
#   sess.run(init)

#   for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
#     _b2_grads, _mean, _stddev, _diff_grads = sess.run([b2_grads, mean, stddev, diff_grads], feed_dict={x: batch_xs, y_: batch_ys})
#     # print(_b2_grads)
#     print("MEAN: {}      STDDEV:{}".format(_mean,_stddev))
#     print('diff grads should be around zero... {}'.format(_diff_grads))
#     exit()
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#     correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     if i % 10 == 0:
#       print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


