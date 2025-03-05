# Policy Gradient method
A practical for policy gradient implementation, training, and evaluation for the CartPole-v1 environment.

## Dependency Management
Use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) to manage the environment and 3rd party libraries.
All the required dependencies are put in requirements.txt.
* Create an environment `conda create -n pg python=3.12`
* Activate the environment `conda activate pg`
* * Install the dependencies 
  * `conda install --yes --file policy_gradient/requirements.txt`

## Training
* Run script directly, `python3 -m policy_gradient.train` or just run train.py script in your IDE.

## References
* https://huggingface.co/learn/deep-rl-course/unit4/introduction