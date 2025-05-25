# Deep Q-Network (DQN)
A practical Deep Q-Network (DQN) implementation, training, and inference for the CartPole-v1 environment.

DQN is a value-based RL method that combines Q-learning (an off-policy algorithm) with deep neural networks to approximate the Q-function, which estimates the expected cumulative reward for taking an action in a given state and following an optimal policy thereafter.

## Dependency Management
Use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) to manage the environment and 3rd party libraries.
All the required dependencies are put in requirements.txt.
* Create an environment `conda create -n dqn python=3.12`
* Activate the environment `conda activate dqn`
  * Install the dependencies 
    * `conda install --yes --file dqn/requirements.txt`

## Training
* Components
  * Training Neural Network: A deep neural network approximates Q(s, a), taking state s as input and outputting Q-values for all possible actions, where Q-value is the expected future cumulative reward the agent will receive by taking action 'a' in state 's' and following the optimal policy from that point on. 
  * Experience Replay: Uses a replay buffer to store experiences (state, action, reward, next state) and sample batches of experiences randomly for training. This helps to reduce correlation between consecutive samples and stabilize training. 
  * Target Neural Network: Employs a target network, which is a copy of the prediction network with slightly delayed updates, to provide more stable training targets
  * Optimizer: Minimizes the TD error between predicted Q-values and target Q-values.
  * Epsilon-Greedy Policy: Uses an ϵ-greedy policy (off-policy) to balance exploration (trying different actions) and exploitation (choosing the best action based on Q-values). 

* Train Script Run
  * Run script directly, `python3 -m dqn.train` or just run train.py script in your IDE.
  * One DQN training run result. It represents the total accumulated reward over each training episode. 
  A training episode in RL refers to a complete sequence of interactions between an agent and its environment, starting from a specific initial state and ending at a defined termination condition.   
  
  ![One training run result](./dqn_train_result.png)

  From the result, we can see that the agent learns to navigate the CartPole-v1 environment effectively.

## Inference
* Run script directly, `python3 -m dqn.inference` or just run inference.py script in your IDE.

## References
* https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
* https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html
* https://huggingface.co/learn/deep-rl-course/unit2/introduction
* https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
* https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
* https://gym.openai.com/