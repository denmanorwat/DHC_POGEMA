from model import Network
import torch
import os
import numpy as np
from pogema import pogema_v0
from pogema import GridConfig
import configs
from DHC_wrapper import DHC_wrapper
import pickle



device = torch.device('cpu')

network = Network()
network.eval()
network.to(device)
path = os.path.dirname(os.path.realpath(__file__))
test_env = "Normal small"

success_rate = 0
mean_steps = []
for test in range(200):
    test_example = path + "/" + test_env + "/" + str(test)
    start_pos, end_pos = np.load(test_example+"/start_pos.npy"), np.load(test_example+"/end_pos.npy")
    with open(test_example + '/' + 'grid.pickle', 'rb') as handle:
        grid = pickle.load(handle)
    print(grid)
    grid_config = GridConfig(on_target="nothing", obs_radius=configs.obs_radius, num_agents=configs.num_agents,
                             max_episode_steps=configs.max_episode_length,
                             agents_xy=start_pos.tolist(), targets_xy=end_pos.tolist(),
                             collision_system="priority", map=grid)
    env = pogema_v0(grid_config)
    env = DHC_wrapper(env)
    state_dict = torch.load('./models/{}.pth'.format("22000"), map_location=device)
    network.load_state_dict(state_dict)
    network.eval()

    obs, pos = env.observe()

    done = False
    network.reset()

    step = 0
    while not done and env.steps < configs.max_episode_length:
        actions, _, _, _ = network.step(torch.as_tensor(obs.astype(np.float32)),
                                        torch.as_tensor(pos.astype(np.float32)))
        (obs, pos), _, done, _ = env.step(actions)
        step += 1
    success_rate = np.array_equal(env.get_agents_xy(), env.get_targets_xy())
    mean_steps.append(step)
print(success_rate, np.array(mean_steps).mean())
