import tensorflow as tf  # version 1.5
import bball
import numpy as np

def normalize_rewards(all_rewards):
    "normalize the rewards to have a std of 1 and mean 0"
    reward_mean = np.mean(all_rewards)
    reward_std =  np.std(all_rewards)
    return [(reward - reward_mean)/reward_std for reward in all_rewards]


if __name__ == '__main__':
# if True:
    # specify nueral network architecture
    n_inputs = 3
    n_hidden = 3
    n_outputs = 1
    # https://github.com/tensorflow/docs/blob/r1.5/site/en/api_docs/python/tf/contrib/layers/variance_scaling_initializer.md
    initializer = tf.contrib.layers.variance_scaling_initializer()

    # build the neural network
    # https://github.com/tensorflow/docs/blob/r1.5/site/en/api_docs/python/tf/placeholder.md
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])

    # https://github.com/tensorflow/docs/blob/r1.5/site/en/api_docs/python/tf/layers/Dense.md
    # activation function: Exponential Linear Units
    # https://github.com/tensorflow/docs/blob/r1.5/site/en/api_docs/python/tf/nn/elu.md
    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                            kernel_initializer=initializer)
    logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
    # ...no documentation... :(
    outputs = tf.nn.sigmoid(logits)

    # map the sigmoid function to be between 0 and 90 degrees
    # https://stackoverflow.com/a/1826651/3585557
    angle = outputs * 90

    # https://github.com/tensorflow/docs/blob/r1.5/site/en/api_docs/python/tf/to_float.md
    y = tf.to_float(angle)

    learning_rate = 0.01
    # https://github.com/tensorflow/docs/blob/r1.5/site/en/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits.md
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                            logits=logits)
    # https://github.com/tensorflow/docs/blob/r1.5/site/en/api_docs/python/tf/train/AdamOptimizer.md
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(cross_entropy)

    gradients = []
    gradient_placeholders = []
    grads_and_vars_feed = []
    for grad, variable in grads_and_vars:
        gradients.append(grad)
        gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, variable))
    training_op = optimizer.apply_gradients(grads_and_vars_feed)

    # https://github.com/tensorflow/docs/blob/r1.5/site/en/api_docs/python/tf/global_variables_initializer.md
    # this needs to come AFTER setting up the neural network and optimizer
    init = tf.global_variables_initializer()

    # https://github.com/tensorflow/docs/blob/r1.5/site/en/api_docs/python/tf/train/Saver.md
    saver = tf.train.Saver()

    n_iterations = 250
    n_shots_per_update = 100
    save_iterations = 10

    # https://github.com/tensorflow/docs/blob/r1.5/site/en/api_docs/python/tf/Session.md
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            all_rewards = []
            all_gradients = []
            for i_shot in range(n_shots_per_update):
                shot = bball.Shot()
                params = shot.setup()
                angle_val, gradients_val = sess.run(
                    [angle, gradients],
                    feed_dict={X: params.reshape(1, n_inputs)}
                )
                reward = shot.shoot(angle_val[0][0])  # not sure about this indexing
                all_rewards.append(reward)
                all_gradients.append(gradients_val)

            all_rewards = normalize_rewards(all_rewards)
            feed_dict = {}
            for i_var, gradient_placeholder in enumerate(gradient_placeholders):
                mean_gradients = np.mean([
                    reward * all_gradients[i_shot][i_var]
                    for i_shot, reward in enumerate(all_rewards)
                ], axis=0)
                feed_dict[gradient_placeholder] = mean_gradients
            sess.run(training_op, feed_dict=feed_dict)
            if iteration % save_iterations == 0:
                saver.save(sess, './my_policy_net_pg.ckpt')

# TODO: use this trained model...