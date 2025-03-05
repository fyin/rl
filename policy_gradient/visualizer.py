import matplotlib.pyplot as plt
import torch

"""
Visualize total accumulated scores over training episodes. This helps analyze how well the agent is learning over time(longer durations → better policy). 
"""
def plot_score(episode_scores=[], title="Result"):
    plt.figure(1)
    scores_tensor = torch.tensor(episode_scores, dtype=torch.float)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total score')
    plt.plot(scores_tensor.numpy())
    # Take 100 episode averages and plot them too
    if len(scores_tensor) >= 100:
        means = scores_tensor.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig("pg_train_result.png")
    plt.show()