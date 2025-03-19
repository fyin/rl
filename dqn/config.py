def get_config():
    return {
        "replay_buffer_size": 10_000,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "epsilon_start": 0.9,
        "epsilon_end": 0.05,
        "epsilon_decay": 1_000,
        "tau": 5e-3,
        "gamma": 0.99,
        "num_episodes": 40_000,
        "tensorboard_path": "run/dqn_tensorboard",
        "model_path": "model/dqn_model"
    }