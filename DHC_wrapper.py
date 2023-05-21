import numpy as np
from pogema.envs import Pogema, PogemaCoopFinish, PogemaLifeLong

INFINITY = 2147483647
def extract_env(pogema_env):
    env = pogema_env
    while(type(env) != type(Pogema()) and type(env) != type(PogemaCoopFinish()) and type(env) != type(PogemaLifeLong())):
        env = env.env
    return env

class DHC_wrapper(Pogema):
    def __init__(self, pogema_env):
        self.pogema_env = extract_env(pogema_env)
        self.pogema_env.reset()
        for i, action in enumerate(self.pogema_env.grid_config.MOVES):
            if action[0] == 0 and action[1] == 0:
                STAY = i
                break
        self.STAY = STAY
        self.num_agents = self.pogema_env.get_num_agents()
        self.map_size = self.pogema_env.grid.obstacles.shape
        self.steps = 0
        self.target_achieved = np.array([False for i in range(self.num_agents)])
        self.obstacles = self.pogema_env.grid.get_obstacles()
        self.aggregated_rewards = []
        self.get_heuri_map()

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return np.array(self.pogema_env.get_agents_xy(only_active=only_active, ignore_borders=ignore_borders))
    
    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return np.array(self.pogema_env.get_targets_xy(only_active=only_active, ignore_borders=ignore_borders))
    
    def get_heuri_map(self):
        num_agents = self.num_agents
        map_size = self.map_size
        dist_map = np.ones((num_agents, *map_size), dtype=np.int32) * INFINITY
        goal_pos = self.get_targets_xy(ignore_borders=False)
        obs_radius = self.pogema_env.grid_config.obs_radius
        
        grid = self.pogema_env.grid
        FREE, OBSTACLE = grid.config.FREE, grid.config.OBSTACLE
        obstacles = grid.obstacles
        
        for i in range(num_agents):
            open_list = list()
            x, y = tuple(goal_pos[i])
            open_list.append((x, y))
            dist_map[i, x, y] = 0

            while open_list:
                x, y = open_list.pop(0)
                dist = dist_map[i, x, y]

                up = x-1, y
                if up[0] >= 0 and obstacles[up]==FREE and dist_map[i, x-1, y] > dist+1:
                    dist_map[i, x-1, y] = dist+1
                    if up not in open_list:
                        open_list.append(up)

                down = x+1, y
                if down[0] < map_size[0] and obstacles[down]==FREE and dist_map[i, x+1, y] > dist+1:
                    dist_map[i, x+1, y] = dist+1
                    if down not in open_list:
                        open_list.append(down)
                
                left = x, y-1
                if left[1] >= 0 and obstacles[left]==FREE and dist_map[i, x, y-1] > dist+1:
                    dist_map[i, x, y-1] = dist+1
                    if left not in open_list:
                        open_list.append(left)
                
                right = x, y+1
                if right[1] < map_size[1] and obstacles[right]==FREE and dist_map[i, x, y+1] > dist+1:
                    dist_map[i, x, y+1] = dist+1
                    if right not in open_list:
                        open_list.append(right)
            
        self.heuri_map = np.zeros((num_agents, 4, *map_size), dtype=np.bool)

        for x in range(map_size[0]):
            for y in range(map_size[1]):
                if obstacles[x, y] == FREE:
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
        self.dist_map = dist_map


    def step(self, actions):
        num_agents = self.num_agents
        obs_radius = self.pogema_env.grid_config.obs_radius
        prev_pos = self.get_agents_xy(ignore_borders=True)
        _, rewards, _, _ = self.pogema_env.step(actions)
        rewards = np.array(rewards)*3
        rewards -= 0.075
        next_pos = self.get_agents_xy(ignore_borders=True)

        goals_pos = self.get_targets_xy(ignore_borders=True)
        agents_pos = np.copy(next_pos)

        self.steps += 1
        goal_achievers = (goals_pos == agents_pos).all(axis=1)
        self.target_achieved = self.target_achieved | goal_achievers

        rewards += goal_achievers*0.075

        collision = ((prev_pos == next_pos).all(axis=1)) & (actions != self.STAY)
        rewards[collision] = -0.5
        self.aggregated_rewards.append(rewards.mean())
        rewards = rewards.tolist()

        if np.array_equal(agents_pos, goals_pos):
            done = True
        else:
            done = False

        info = {'step': self.steps-1}

        # make sure no overlapping agents
        # print("Agents positions: {}".format(agents_pos.shape))
        # print("Number of agents: {}".format(num_agents))
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
        agent_pos = self.get_agents_xy(ignore_borders=True)

        obs = np.zeros((num_agents, 6, 2*obs_radius+1, 2*obs_radius+1), dtype=np.bool)
        obstacles = self.pogema_env.grid.get_obstacles()


        agent_map = np.zeros(map_size, dtype=np.bool)
        agent_map[agent_pos[:, 0], agent_pos[:, 1]] = 1
        for i, agent_position in enumerate(agent_pos):
            x, y = agent_position

            obs[i, 0] = agent_map[x:x+2*obs_radius+1, y:y+2*obs_radius+1]
            obs[i, 0, obs_radius, obs_radius] = 0
            obs[i, 1] = obstacles[x:x+2*obs_radius+1, y:y+2*obs_radius+1]
            obs[i, 2:] = self.heuri_map[i, :, x:x+2*obs_radius+1, y:y+2*obs_radius+1]
        
        return obs, np.copy(agent_pos)
    
    def render(self, mode='human'):
        return self.pogema_env.render(mode)

    def quantity_of_achieved_goals(self):
        return self.target_achieved.sum()

    def mission_complete(self):
        return (self.get_agents_xy() == self.get_targets_xy()).all().sum()

    def get_mean_reward(self):
        return np.array(self.aggregated_rewards).mean()
    
    def reset(self, seed):
        self.steps = 0
        self.pogema_env.reset()
        self.num_agents = self.pogema_env.get_num_agents()
        self.target_achieved = np.array([False for i in range(self.num_agents)])
        self.aggregated_rewards = []
        self.obstacles = self.pogema_env.grid.get_obstacles()
        self.map_size = self.obstacles.shape
        self.get_heuri_map()
        obs, pos = self.observe()
        return obs, pos
    