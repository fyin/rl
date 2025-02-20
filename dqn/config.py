def get_config():
    return {
        "replay_buffer_size": 10000,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "epsilon_start": 0.9,
        "epsilon_end": 0.05,
        "epsilon_decay": 1000,
        "tau": 0.005,
        "gamma": 0.99,
        "num_episodes": 200,
        "tensorboard_path": "run/dqn_tensorboard",
        "model_path": "model/dqn_model"
    }