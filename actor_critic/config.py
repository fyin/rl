def get_config():
    return {
        "learning_rate": 1e-3,
        "epsilon": 0.2,
        "gamma": 0.99,
        "lambda": 0.95,
        "epochs": 10,
        "batch_size": 128,
        "update_nums": 1000,
        "max_terminal_steps": 1000,
    }