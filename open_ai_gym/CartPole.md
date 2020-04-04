# CartPole Example

```python
>>> import gym
>>> env = gym.make("CartPole-v0")
>>> obs = env.reset()
>>> obs
array([-0.03449988, -0.04597018, -0.0213953 ,  0.04740928])
# [horizontal position, velocity, angle of the pole, angular velocity]
>>> env.render() # show result
```

Move the pole to the left (0) or right (1)?

```python
>>> action = 1  # accelerate right
>>> obs, reward, done, info = env.step(action)
>>> obs
array([-0.03541929,  0.14945193, -0.02044711, -0.25194653])
```

## Hardcoded Policy

As a simple approach, lets use a policy to accelerate left when the pole is leaning toward the left and accelerate right when the pole is leaning toward the right.

```python
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_reward = 0
    obs = env.reset()
    for step in range(1000):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            break
    totals.append(episode_reward)
```

With this simple approach, the average steps that the pole stayed up was roughly 42 with a max of 63 and a min of 24.

```python
In [25]: import numpy as np
In [26]: np.mean(totals)
Out[26]: 42.032
In [27]: np.std(totals)
Out[27]: 8.687633509765476

In [28]: np.min(totals)
Out[28]: 24.0

In [29]: np.max(totals)
Out[29]: 63.0
```

We can model this using a simple neural network that takes four observations, the inputs for the current state, then outputs the probability of moving left, action 0.

```python
import tensorflow as tf

n_inputs = 4
n_hidden = 4
n_outputs = 4
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)
init = tf.global_variables_initializer()

y = 1.0 - tf.to_float(action)
learning_rate = 0.01
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)

gradients = [grad for grad, variable in grads_and_vars]

gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
```

We need to define the reward function.  Because of the nature of this scenario, the consequence of the action is somewhat delayed from when the action actually takes place.  So we want to reward each action by a weighted sum of all the following results, so if the stick stays up longer after an action, the reward is larger.

```python
def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards
```

Also, we want to normalize the reward so the standard deviation is one and the mean is zero.

```python
def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]
```

This is everything we need to begin training the policy (or model).

```python
n_iterations = 250
n_max_steps = 1000
n_games_per_update = 10
save_iterations = 10
discount_rate = 0.95

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run(
                    [action, gradients],
                    feed_dict={X: obs.reshape(1, n_inputs)}
                )
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}
        for var_index, grad_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean(
                [reward * all_gradients[game_index][step][var_index]
                    for game_index, rewards in enumerate(all_rewards)
                    for step, reward in enumerate(rewards)],
                axis=0)
            feed_dict[grad_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, './my_policy_net_pg.ckpt')
```

