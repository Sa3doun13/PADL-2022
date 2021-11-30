import random
import numpy as np
import wandb
import gym

from JSS.dispatching_rules.JSSEnv import JssEnv
from config import default_config


def RANDOM_worker(default_config):
    wandb.init(config=default_config, name="RANDOM")
    config = wandb.config
    env = JssEnv({'instance_path': config['instance_path']})
    done = False
    state = env.reset()
    while not done:
        real_state = np.copy(state['real_obs'])
        legal_actions = state['action_mask'][:-1]
        index_legal_action = np.where(legal_actions)[0]
        random_action = np.random.choice(index_legal_action, 1, replace=False)[0]
        assert legal_actions[random_action]
        state, reward, done, _ = env.step(random_action)
    print(sum(env.solution[:, 0] == 0))
    env.reset()
    make_span = env.last_time_step
    wandb.log({"nb_episodes": 1, "make_span": make_span})


if __name__ == "__main__":
    RANDOM_worker(default_config)
