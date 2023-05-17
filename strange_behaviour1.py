from pogema.grid_config import GridConfig
from pogema import pogema_v0

test_config = GridConfig(size=3, density=0, num_agents=2, obs_radius=2, agents_xy=[[1, 0], [0, 0]], targets_xy=[[2, 2], [2, 1]],
                          collision_system="block_both")
env = pogema_v0(test_config)
env.reset()
env.render()
env.step([2, 2])
env.render()

test_config = GridConfig(size=3, density=0, num_agents=2, obs_radius=2, agents_xy=[[0, 0], [1, 0]], targets_xy=[[2, 2], [2, 1]],
                          collision_system="block_both")
env = pogema_v0(test_config)
env.reset()
env.render()
env.step([2, 2])
print(env.grid.get_obstacles())

env.render()