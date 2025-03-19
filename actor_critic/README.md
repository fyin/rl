# Actor Critic
A practical PPO (one Actor Critic) reinforcement learning implementation and training for the CartPole-v1 environment.

## Dependency Management
Use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) to manage the environment and 3rd party libraries.
All the required dependencies are put in requirements.txt.
* Create an environment `conda create -n ac python=3.12`
* Activate the environment `conda activate ac`
* * Install the dependencies 
  * `conda install --yes --file actor_critic/requirements.txt`

## Training
* Run script directly, `python3 -m actor_critic.train` or just run train.py script in your IDE.
* ![One training run result](./ppo_train_result.png)
## References
* https://huggingface.co/learn/deep-rl-course/unit8/hands-on-cleanrl