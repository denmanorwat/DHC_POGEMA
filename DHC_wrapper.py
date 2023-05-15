import numpy as np
from pogema.envs import Pogema

class DHC_wrapper(Pogema):

    def __init__(self, pogema_env):
        self.pogema_env = pogema_env
        for i, action in enumerate(self.pogema_env.grid.config.MOVES):
            if action[0] == 0 and action[1] == 0:
                STAY = i
                break
        self.STAY = STAY
        self.num_agents = self.pogema_env.get_num_agents()
        self.map_size = self.pogema_env.grid_config.size
        self.steps = 0
    
    def get_heuri_map(self):
        num_agents = self.num_agents
        map_size = self.map_size
        dist_map = np.ones((num_agents, *map_size), dtype=np.int32) * 2147483647
        goal_pos = self.pogema_env.grid_config.get_targets_xy()
        obstacles = self.pogema_env.grid.get_obstacles()
        obs_radius = self.pogema_env.grid_config.obs_radius

        
        for i in range(num_agents):
            open_list = list()
            x, y = tuple(goal_pos[i])
            open_list.append((x, y))
            dist_map[i, x, y] = 0

            while open_list:
                x, y = open_list.pop(0)
                dist = dist_map[i, x, y]

                up = x-1, y
                if up[0] >= 0 and obstacles[up]=="#" and dist_map[i, x-1, y] > dist+1:
                    dist_map[i, x-1, y] = dist+1
                    if up not in open_list:
                        open_list.append(up)

                down = x+1, y
                if down[0] < map_size[0] and obstacles[down]=="#" and dist_map[i, x+1, y] > dist+1:
                    dist_map[i, x+1, y] = dist+1
                    if down not in open_list:
                        open_list.append(down)
                
                left = x, y-1
                if left[1] >= 0 and obstacles[left]=="#" and dist_map[i, x, y-1] > dist+1:
                    dist_map[i, x, y-1] = dist+1
                    if left not in open_list:
                        open_list.append(left)
                
                right = x, y+1
                if right[1] < map_size[1] and obstacles[right]=="#" and dist_map[i, x, y+1] > dist+1:
                    dist_map[i, x, y+1] = dist+1
                    if right not in open_list:
                        open_list.append(right)
            
        self.heuri_map = np.zeros((num_agents, 4, *map_size), dtype=np.bool)

        for x in range(map_size[0]):
            for y in range(map_size[1]):
                if obstacles[x, y] == "#":
                    for i in range(num_agents):

                        if x > 0 and dist_map[i, x-1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x-1, y] == dist_map[i, x, y]-1
                            self.heuri_map[i, 0, x, y] = 1
                        
                        if x < map_size[0]-1 and dist_map[i, x+1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x+1, y] == dist_map[i, x, y]-1
                            self.heuri_map[i, 1, x, y] = 1

                        if y > 0 and dist_map[i, x, y-1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y-1] == dist_map[i, x, y]-1
                            self.heuri_map[i, 2, x, y] = 1
                        
                        if y < map_size[1]-1 and dist_map[i, x, y+1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y+1] == dist_map[i, x, y]-1
                            self.heuri_map[i, 3, x, y] = 1
        
        self.heuri_map = np.pad(self.heuri_map, ((0, 0), (0, 0), (obs_radius, obs_radius), (obs_radius, obs_radius)))


    def step(self, actions):
        num_agents = self.num_agents
        obs_radius = self.pogema_env.grid_config.obs_radius
        prev_pos = self.pogema_env.get_agents_xy()
        _, rewards, _, _ = self.pogema_env.step(actions)
        next_pos = self.pogema_env.get_agents_xy()

        goals_pos = self.pogema_env.get_targets_xy()
        agents_pos = np.copy(next_pos)

        self.steps += 1
        rewards = rewards*3
        rewards = rewards-0.075

        collision = (prev_pos == next_pos) & (actions!=self.STAY)
        print("Collisions: {}".format(collision))
        rewards[collision] = -0.5
        print("Rewards: {}".format(rewards))
        

        if np.array_equal(agents_pos, goals_pos):
            done = True
        else:
            done = False

        info = {'step': self.steps-1}

        # make sure no overlapping agents
        if np.unique(agents_pos, axis=0).shape[0] < num_agents:
            raise RuntimeError('unique')

        # update last actions
        self.last_actions = np.zeros((num_agents, 5, 2*obs_radius+1, 2*obs_radius+1), dtype=np.bool)
        self.last_actions[np.arange(num_agents), np.array(actions)] = 1

        return self.observe(), rewards, done, info

    def observe(self):
        num_agents = self.num_agents
        obs_radius = self.pogema_env.grid_config.obs_radius
        map_size = self.map_size
        agent_pos = self.pogema_env.get_agents_xy()

        obs = np.zeros((num_agents, 6, 2*obs_radius+1, 2*obs_radius+1), dtype=np.bool)
        obstacles = self.pogema_env.grid.get_obstacles()

        obstacle_map = np.pad(obstacles, obs_radius, 'constant', constant_values=0)

        agent_map = np.zeros((map_size), dtype=np.bool)
        agent_map[agent_pos[:,0], agent_pos[:,1]] = 1
        agent_map = np.pad(agent_map, obs_radius, 'constant', constant_values=0)

        for i, agent_pos in enumerate(agent_pos):
            x, y = agent_pos

            obs[i, 0] = agent_map[x:x+2*obs_radius+1, y:y+2*obs_radius+1]
            obs[i, 0, obs_radius, obs_radius] = 0
            obs[i, 1] = obstacle_map[x:x+2*obs_radius+1, y:y+2*obs_radius+1]
            obs[i, 2:] = self.heuri_map[i, :, x:x+2*obs_radius+1, y:y+2*obs_radius+1]
        
        return obs, np.copy(agent_pos)
    
    def render(self, mode='human'):
        return self.pogema_env.render(mode)
    
    def reset(self, seed):
        self.steps=0
        self.pogema_env.reset(seed)
    
    