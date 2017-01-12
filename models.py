import tensorflow as tf



def calculate_grad(var, err):
  return tf.gradients(err, var)[0]

def update_with_grad(var, grad, lr):
  return var - (lr*grad)

def calculate_grad_and_updated_elem(var, err, lr):
  grad = calculate_grad(var, err)
  updated = update_with_grad(var, grad, lr)
  return grad, updated

def create_updated_elem(var, err, lr):
  grad = tf.gradients(err, var)[0]
  new_tensor = var - (lr*grad)
  return new_tensor

def get_squared_sum_of_list_of_parameters(array_of_params):
  squares = [tf.reduce_sum(tf.square(p)) for p in array_of_params]
  return tf.add_n(squares)

def calculate_second_derivative(err_diff, lr, grad_size_squared):
  # This is absolute valued to death. I can simplify this, for sure.... Because now I need the sign!
  # return 2*tf.abs((tf.abs(err_diff) - tf.abs(lr*grad_size_squared)) / (lr*lr*grad_size_squared)) #check this out later...
  # return 2*(err_diff - lr*grad_size_squared) / (lr*lr*grad_size_squared) #check this out later...
  return -2*(-err_diff - lr*grad_size_squared) / (lr*lr*grad_size_squared) #check this out later...


def one_layer_model(test_lr, max_lr, sigmoid_scale=1.0, min_lr=0.0001):
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])

  x_64 = tf.to_double(x)
  y_64 = tf.to_double(y_)

  W1 = tf.get_variable("W1", shape=[784, 10], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float64))
  b1 = tf.Variable(tf.zeros([10], dtype=tf.float64), dtype=tf.float64)
  unscaled_y = tf.matmul(x_64,W1) + b1
  y = tf.nn.softmax(unscaled_y)

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(unscaled_y, y_64))

  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_64,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  W1_grad, similar_W1 = calculate_grad_and_updated_elem(W1, cross_entropy, test_lr)
  b1_grad, similar_b1 = calculate_grad_and_updated_elem(b1, cross_entropy, test_lr)

  updated_unscaled_y = tf.matmul(x_64, similar_W1) + similar_b1
  updated_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(updated_unscaled_y, y_64))

  grad_size_squared = get_squared_sum_of_list_of_parameters([W1_grad, b1_grad])

  ce_diff = updated_cross_entropy - cross_entropy

  second_derivative = calculate_second_derivative(ce_diff, test_lr, grad_size_squared)

  second_derivative_scaled = tf.mul(second_derivative, sigmoid_scale)

  second_derivative_sigmoided = tf.nn.sigmoid(second_derivative)

  true_LR = tf.maximum(max_lr * second_derivative_sigmoided, min_lr)

  update_W1 = tf.assign(W1, W1 - true_LR*W1_grad)
  update_b1 = tf.assign(b1, b1 - true_LR*b1_grad)

  updates = tf.group(update_W1, update_b1)
  # updates = tf.train.GradientDescentOptimizer(max_lr/2).minimize(cross_entropy)

  var_list= [W1, b1]

  return {
    'input' : x,
    'target_output' : y_,
    'cross_entropy' : cross_entropy,
    'ce_diff' : ce_diff,
    'accuracy' : accuracy,
    'var_list' : var_list,
    'second_derivative' : second_derivative,
    'second_derivative_sigmoided' : second_derivative_sigmoided,
    'grad_size_squared' : grad_size_squared,
    'updates' : updates
  }


def two_layer_model(test_lr, max_lr, sigmoid_scale=1.0):
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])

  x_64 = tf.to_double(x)
  y_64 = tf.to_double(y_)

  W1 = tf.get_variable("W1", shape=[784, 200], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float64))
  b1 = tf.Variable(tf.zeros([200], dtype=tf.float64), dtype=tf.float64)
  h1 = tf.nn.relu(tf.matmul(x_64, W1) + b1)

  W2 = tf.get_variable("W2", shape=[200, 10], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float64))
  b2 = tf.Variable(tf.zeros([10], dtype=tf.float64), dtype=tf.float64)
  unscaled_y = tf.matmul(h1,W2) + b2
  y = tf.nn.softmax(unscaled_y)

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(unscaled_y, y_64))

  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_64,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  W1_grad, similar_W1 = calculate_grad_and_updated_elem(W1, cross_entropy, test_lr)
  b1_grad, similar_b1 = calculate_grad_and_updated_elem(b1, cross_entropy, test_lr)
  W2_grad, similar_W2 = calculate_grad_and_updated_elem(W2, cross_entropy, test_lr)
  b2_grad, similar_b2 = calculate_grad_and_updated_elem(b2, cross_entropy, test_lr)

  updated_h1 = tf.matmul(x_64, similar_W1) + similar_b1
  updated_unscaled_y = tf.matmul(updated_h1, similar_W2) + similar_b2
  updated_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(updated_unscaled_y, y_64))

  grad_size_squared = get_squared_sum_of_list_of_parameters([W1_grad, b1_grad, W2_grad, b2_grad])

  ce_diff = updated_cross_entropy - cross_entropy

  second_derivative = calculate_second_derivative(ce_diff, test_lr, grad_size_squared)

  second_derivative_scaled = tf.mul(second_derivative, sigmoid_scale)

  second_derivative_sigmoided = tf.nn.sigmoid(second_derivative)

  true_LR = max_lr * second_derivative_sigmoided

  update_W1 = tf.assign(W1, W1 - true_LR*W1_grad)
  update_b1 = tf.assign(b1, b1 - true_LR*b1_grad)
  update_W2 = tf.assign(W2, W2 - true_LR*W2_grad)
  update_b2 = tf.assign(b2, b2 - true_LR*b2_grad)  

  updates = tf.group(update_W1, update_b1, update_W2, update_b2)
  # updates = tf.train.GradientDescentOptimizer(max_lr/2).minimize(cross_entropy)

  var_list= [W1, b1]

  return {
    'input' : x,
    'target_output' : y_,
    'cross_entropy' : cross_entropy,
    'ce_diff' : ce_diff,
    'accuracy' : accuracy,
    'var_list' : var_list,
    'second_derivative' : second_derivative,
    'second_derivative_sigmoided' : second_derivative_sigmoided,
    'grad_size_squared' : grad_size_squared,
    'updates' : updates
  }



