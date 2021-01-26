
Batch size does indeed mean the same thing in reinforcement learning, compared to supervised learning. The intuition of "batch learning" (usually in mini-batch) is two-fold:
    Due to memory constraints of hardware, it may be difficult to do batch gradient descent on over 1,000,000 data points.
    To calculate the gradient of loss on an subset of the whole data, that is representative of the whole data. If the batch you train on at each step is not representative of the whole data, there will be bias in your update step.
In supervised learning, such as neural networks, you would do mini-batch gradient descent to update your neural network. In deep reinforcement learning, you're training the same neural networks, so it works in the same way.
In supervised learning, your batch would consist of a set of features, and its respective labels. In deep reinforcement learning, it is similar. It is a tuple (state, action, reward, state at t + 1, sometimes done).
State: The original state that describes your environment
Action: The action you performed in that environmental state
Reward: Reward signal obtained after performing that action in that state
State t+1: The new state your action transitioned you to.
Done: A boolean referring to the end of your task. For example, if you train RL to play chess, done would be either winning or losing the chess game.
You would sample a batch of these (s, a, r, s(t+1), done) tuples. Then you feed it into the TD update rule, usually in the form of:
enter image description here
The two Q's are the action values, and are calculated by passing s, s(t+1) and a into your neural network.
Then, you would update your neural network with the Q as the label.
