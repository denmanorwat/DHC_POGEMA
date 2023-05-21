import os
import numpy as np
import pickle
from pogema import pogema_v0
from pogema import Normal8x8, Hard8x8, Normal16x16, Hard16x16

environments = {"Normal small": Normal8x8, "Hard small": Hard8x8,
                "Normal medium": Normal16x16, "Hard medium": Hard16x16}

def numpy_to_ascii(obstacles):
    ascii_art = ""
    for row in obstacles:
        for elem in row:
            if elem == 0:
                ascii_art = ''.join((ascii_art, "."))
            else:
                ascii_art = ''.join((ascii_art, "#"))
        ascii_art = ''.join((ascii_art, '\n'))
    print(ascii_art)
    return ascii_art

for name, CONSTRUCTOR in environments.items():
    path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(path+"/"+name):
        os.mkdir(path+"/"+name)
    for i in range(200):
        target_dir = path + "/" + name + "/" + str(i)
        if os.path.exists(target_dir):
            os.rmdir(target_dir)
        os.mkdir(target_dir)
        config = CONSTRUCTOR()
        env = pogema_v0()
        env.reset()
        start_pos, end_pos = np.array(env.get_agents_xy(ignore_borders=True)),\
                             np.array(env.get_targets_xy(ignore_borders=True))
        obstacles = env.grid.obstacles
        obstacles = obstacles[config.obs_radius:-config.obs_radius, config.obs_radius:-config.obs_radius]
        grid = numpy_to_ascii(obstacles)
        np.save(target_dir+"/start_pos", start_pos)
        np.save(target_dir+"/end_pos", end_pos)
        with open(target_dir + '/' + 'grid.pickle', 'wb') as handle:
            pickle.dump(grid, handle)
