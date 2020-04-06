import tensorflow as tf  # version 1.5
import bball
import numpy as np

# https://github.com/tensorflow/docs/blob/r1.5/site/en/api_docs/python/tf/

def normalize_rewards(all_rewards):
    "normalize the rewards to have a std of 1 and mean 0"
    reward_mean = np.mean(all_rewards)
    reward_std =  np.std(all_rewards)
    return [(reward - reward_mean)/reward_std for reward in all_rewards]


def main():
    NotImplemented

if __name__ == '__main__':
# if True:
    # ~~~Specify neural network architecture~~~
    n_inputs = 3
    n_hidden = 3
    n_outputs = 1

    n_iterations = 2000000
    n_shots_per_update = 500
    save_iterations = 10000

    initializer = tf.variance_scaling_initializer()

    # ~~~Build the neural network~~~
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    y = tf.placeholder(tf.float32, shape=[None, n_outputs])

    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                            kernel_initializer=initializer)

    # get an output angle to shoot at
    output = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
    # by not providing an activation function to this second layer, it defaults
    # to a 'linear' activation, or no activation. This is appropriate for
    # continuous output, across all possible real numbers.

    y = tf.to_float(output)  # treat output angle as best possible option
    learning_rate = 0.01
    loss = tf.reduce_mean(tf.square(output - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    training_op = optimizer.apply_gradients(grads_and_vars_feed)

    # this needs to come AFTER setting up the neural network and optimizer
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            all_rewards = []
            all_gradients = []
            for i_shot in range(n_shots_per_update):
                shot = bball.Shot()
                params = shot.setup()
                angle_val, gradients_val = sess.run(
                    [output, gradients],
                    feed_dict={X: params.reshape(1, n_inputs)}
                )
                reward = shot.shoot(angle_val[0][0])  # negate the reward
                if reward > 20:
                    print(angle_val[0][0], reward)
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
                saver.save(sess, 'debug_04/policy_net_pg.ckpt')
