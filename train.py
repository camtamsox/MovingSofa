import math
import numpy as np
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import random
import pickle
from gym import Env
from gym.spaces import Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import time

def initialize_circle(num_vertices):
    assert num_vertices%2 == 0

    radius = 0.3

    vertices = np.zeros(num_vertices*2) #create x (even) and y (odd)

    radius_squared = radius * radius

    x_increment = 2*radius/((num_vertices/2)-1) # ensure circle has domain [-radius*2,0]
    # upper half
    for j in range(0, int(num_vertices), 2):
        # x value
        vertices[j] = j * x_increment/2.0001

        # y value
        vertices[j + 1] = 0.98*(math.sqrt(radius_squared - pow(vertices[j] - radius, 2))) #0.99 prevents rounding errors

    # lower half
    for j in range(0, int(num_vertices), 2):
        # x value
        vertices[j + int(num_vertices)] = (2*radius-x_increment*j/2.0001)

        # y value
        vertices[j + int(num_vertices) + 1] = 0.98*(-math.sqrt(radius_squared - pow(vertices[j + int(num_vertices)] - radius, 2))) #0.99 prevents rounding errors

    for i in range(0,int(num_vertices*2),2):
        vertices[i] = vertices[i] - 1

    # put into list
    shape_list = []
    for i in range(0, int(num_vertices*2), 2):
        shape_list.append((vertices[i] - 1.5, vertices[i+1])) # -1 to shift to left
    shape = Polygon(shape_list)
    return shape


radius = math.sqrt(2)/2
hallway_length = 4
def transform_hallway(theta_total_change, x_total_change, y_total_change):
    # can optimize by doing common number thing

    slope_horizontal = (radius * math.sin(theta_total_change + 3*math.pi/4) - radius * math.sin(theta_total_change + math.pi/4))/(radius * math.cos(theta_total_change + 3*math.pi/4) - radius * math.cos(theta_total_change + math.pi/4))
    slope_vertical = (radius * math.sin(theta_total_change + 5*math.pi/4) - radius * math.sin(theta_total_change + 3*math.pi/4))/(radius * math.cos(theta_total_change + 5*math.pi/4) - radius * math.cos(theta_total_change + 3*math.pi/4))
  
    # left upper
    if theta_total_change<=math.pi/2 or theta_total_change>=3*math.pi/2:
        x_left_upper = -(2 * radius * slope_horizontal * math.sin(math.pi/4 + theta_total_change) - 2 * radius * slope_horizontal * slope_horizontal * math.cos(math.pi/4 + theta_total_change))/(2 * (slope_horizontal * slope_horizontal + 1)) - math.sqrt((2 * radius*slope_horizontal*math.sin(math.pi/4 + theta_total_change) - 2*radius*slope_horizontal* slope_horizontal*math.cos(math.pi/4 + theta_total_change))**2/(4 * (slope_horizontal**2 +1)**2) - 1/(slope_horizontal**2 + 1) * (-hallway_length**2 + radius**2 * slope_horizontal**2 * math.cos(math.pi/4 + theta_total_change)**2 - 2*radius**2 * slope_horizontal * math.cos(math.pi/4 + theta_total_change)* math.sin(math.pi/4 + theta_total_change) + radius**2 * math.sin(math.pi/4 + theta_total_change)**2))
    else:
        x_left_upper = -(2 * radius * slope_horizontal * math.sin(math.pi/4 + theta_total_change) - 2 * radius * slope_horizontal * slope_horizontal * math.cos(math.pi/4 + theta_total_change))/(2 * (slope_horizontal * slope_horizontal + 1)) + math.sqrt((2 * radius*slope_horizontal*math.sin(math.pi/4 + theta_total_change) - 2*radius*slope_horizontal* slope_horizontal*math.cos(math.pi/4 + theta_total_change))**2/(4 * (slope_horizontal**2 +1)**2) - 1/(slope_horizontal**2 + 1) * (-hallway_length**2 + radius**2 * slope_horizontal**2 * math.cos(math.pi/4 + theta_total_change)**2 - 2*radius**2 * slope_horizontal * math.cos(math.pi/4 + theta_total_change)* math.sin(math.pi/4 + theta_total_change) + radius**2 * math.sin(math.pi/4 + theta_total_change)**2))
  
    y_left_upper = slope_horizontal*(x_left_upper - radius * math.cos(theta_total_change + math.pi/4)) + radius*math.sin(theta_total_change + math.pi/4)

    # left lower
    if theta_total_change <= math.pi/2 or theta_total_change>= 3*math.pi/2:
        x_left_lower = -(2 * radius * slope_horizontal**2 * math.cos(math.pi/4 + theta_total_change) - 2 * radius * slope_horizontal * math.sin(math.pi/4 + theta_total_change))/(2 * (slope_horizontal * slope_horizontal + 1)) - math.sqrt((2 * radius*slope_horizontal**2 * math.cos(math.pi/4 + theta_total_change) - 2*radius*slope_horizontal * math.sin(math.pi/4 + theta_total_change))**2/(4 * (slope_horizontal**2 +1)**2)- 1/(slope_horizontal**2 + 1) * (-hallway_length**2 + radius**2 * slope_horizontal**2 * math.cos(math.pi/4 + theta_total_change)**2 - 2 * radius**2 * slope_horizontal * math.cos(math.pi/4 + theta_total_change)*math.sin(math.pi/4 + theta_total_change)+radius**2 * math.sin(math.pi/4 + theta_total_change)**2))
    else:
        x_left_lower = -(2 * radius * slope_horizontal**2 * math.cos(math.pi/4 + theta_total_change) - 2 * radius * slope_horizontal * math.sin(math.pi/4 + theta_total_change))/(2 * (slope_horizontal * slope_horizontal + 1)) + math.sqrt((2 * radius*slope_horizontal**2 * math.cos(math.pi/4 + theta_total_change) - 2*radius*slope_horizontal * math.sin(math.pi/4 + theta_total_change))**2/(4 * (slope_horizontal**2 +1)**2)- 1/(slope_horizontal**2 + 1) * (-hallway_length**2 + radius**2 * slope_horizontal**2 * math.cos(math.pi/4 + theta_total_change)**2 - 2 * radius**2 * slope_horizontal * math.cos(math.pi/4 + theta_total_change)*math.sin(math.pi/4 + theta_total_change)+radius**2 * math.sin(math.pi/4 + theta_total_change)**2))

    y_left_lower = slope_horizontal*(x_left_lower - radius * math.cos(theta_total_change + 5 * math.pi/4)) + radius*math.sin(theta_total_change + 5 * math.pi/4)

    # middle lower
    if math.pi<= theta_total_change:
        x_middle_lower = -(2 * radius * slope_vertical**2 * math.cos(math.pi/4 + theta_total_change) - 2 * radius * slope_vertical * math.sin(math.pi/4 + theta_total_change))/(2 * (slope_vertical**2 + 1)) - math.sqrt((2 * radius*slope_vertical**2 * math.cos(math.pi/4 + theta_total_change) - 2*radius*slope_vertical * math.sin(math.pi/4 + theta_total_change))**2/(4 * (slope_vertical**2 +1)**2)- 1/(slope_vertical**2 + 1) * (-hallway_length**2 + radius**2 * slope_vertical**2 * math.cos(math.pi/4 + theta_total_change)**2 - 2 * radius**2 * slope_vertical * math.cos(math.pi/4 + theta_total_change)*math.sin(math.pi/4 + theta_total_change)+radius**2 * math.sin(math.pi/4 + theta_total_change)**2))
    else:
        x_middle_lower = -(2 * radius * slope_vertical**2 * math.cos(math.pi/4 + theta_total_change) - 2 * radius * slope_vertical * math.sin(math.pi/4 + theta_total_change))/(2 * (slope_vertical**2 + 1)) + math.sqrt((2 * radius*slope_vertical**2 * math.cos(math.pi/4 + theta_total_change) - 2*radius*slope_vertical * math.sin(math.pi/4 + theta_total_change))**2/(4 * (slope_vertical**2 +1)**2)- 1/(slope_vertical**2 + 1) * (-hallway_length**2 + radius**2 * slope_vertical**2 * math.cos(math.pi/4 + theta_total_change)**2 - 2 * radius**2 * slope_vertical * math.cos(math.pi/4 + theta_total_change)*math.sin(math.pi/4 + theta_total_change)+radius**2 * math.sin(math.pi/4 + theta_total_change)**2))
 
    y_middle_lower = slope_vertical*(x_middle_lower - radius * math.cos(theta_total_change + 5*math.pi/4)) + radius * math.sin(theta_total_change + 5 * math.pi/4)

    # right lower
    if math.pi <= theta_total_change:
        x_right_lower = -(2 * radius * slope_vertical* math.sin(-math.pi/4 + theta_total_change) - 2 * radius * slope_vertical**2 * math.sin(math.pi/4 + theta_total_change))/(2 * (slope_vertical**2 + 1)) - math.sqrt((2 * radius*slope_vertical * math.sin(-math.pi/4 + theta_total_change) - 2*radius*slope_vertical**2 * math.sin(math.pi/4 + theta_total_change))**2/(4 * (slope_vertical**2 +1)**2)- 1/(slope_vertical**2 + 1) * (-hallway_length**2 + radius**2 * math.sin(-math.pi/4 + theta_total_change)**2 - 2 * radius**2 * slope_vertical * math.sin(-math.pi/4 + theta_total_change)*math.sin(math.pi/4 + theta_total_change)+radius**2 * slope_vertical**2 * math.sin(math.pi/4 + theta_total_change)**2))
    else:
        x_right_lower = -(2 * radius * slope_vertical* math.sin(-math.pi/4 + theta_total_change) - 2 * radius * slope_vertical**2 * math.sin(math.pi/4 + theta_total_change))/(2 * (slope_vertical**2 + 1)) + math.sqrt((2 * radius*slope_vertical * math.sin(-math.pi/4 + theta_total_change) - 2*radius*slope_vertical**2 * math.sin(math.pi/4 + theta_total_change))**2/(4 * (slope_vertical**2 +1)**2)- 1/(slope_vertical**2 + 1) * (-hallway_length**2 + radius**2 * math.sin(-math.pi/4 + theta_total_change)**2 - 2 * radius**2 * slope_vertical * math.sin(-math.pi/4 + theta_total_change)*math.sin(math.pi/4 + theta_total_change)+radius**2 * slope_vertical**2 * math.sin(math.pi/4 + theta_total_change)**2))
  
    y_right_lower = slope_vertical * (x_right_lower - radius * math.cos(theta_total_change + 7*math.pi/4)) + radius * math.sin(theta_total_change + 7*math.pi/4)

    # right upper
    x_right_upper = radius * math.cos(theta_total_change + math.pi/4)
    y_right_upper = radius * math.sin(theta_total_change + math.pi/4)


    # middle middle
    x_middle_middle = radius * math.cos(theta_total_change + 5*math.pi/4)
    y_middle_middle = radius * math.sin(theta_total_change + 5*math.pi/4)

    # middle upper
    x_middle_upper = radius * math.cos(theta_total_change + 3*math.pi/4)
    y_middle_upper = radius * math.sin(theta_total_change + 3*math.pi/4)

    hallway_points_list = []
    hallway_points_list.append((x_left_upper - x_total_change, y_left_upper - y_total_change))
    hallway_points_list.append((x_left_lower - x_total_change, y_left_lower - y_total_change))
    hallway_points_list.append((x_middle_middle - x_total_change, y_middle_middle - y_total_change))
    hallway_points_list.append((x_middle_lower - x_total_change, y_middle_lower - y_total_change))
    hallway_points_list.append((x_right_lower - x_total_change, y_right_lower - y_total_change))
    hallway_points_list.append((x_right_upper - x_total_change, y_right_upper - y_total_change))

    hallway_shape = Polygon(hallway_points_list)

    hallway_is_done_points_list = []
    hallway_is_done_points_list.append((x_middle_upper - x_total_change, y_middle_upper - y_total_change))
    hallway_is_done_points_list.append((x_middle_lower - x_total_change, y_middle_lower - y_total_change))
    hallway_is_done_points_list.append((x_right_lower - x_total_change, y_right_lower - y_total_change))
    hallway_is_done_points_list.append((x_right_upper - x_total_change, x_right_upper - y_total_change))

    hallway_is_done_shape = Polygon(hallway_is_done_points_list)

    return hallway_shape, hallway_is_done_shape

def draw_points_and_boundaries(shape_poly, hallway_shape):
    x,y = shape_poly.exterior.xy
    plt.plot(x,y)

    x,y = hallway_shape.exterior.xy
    plt.plot(x,y)
    plt.show()
    return

def check_points(shape_poly, hallway_shape, hallway_is_done_shape):
  # check if all points are in hallway
    if hallway_shape.contains(shape_poly) == False:
        return False, False

  # check if all points are through hallway
    if hallway_is_done_shape.contains(shape_poly) == False:
        return True, False
    
    return True, True
def save_shape(x,y,num_vertices, shape_num):
    data = {
        'x': x,
        'y': y,
        'num_vertices': num_vertices               
        }

    with open(str(shape_num) + '.pickle','wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_shape(shape_num):
    with open(str(shape_num) + '.pickle','rb') as handle:
        data = pickle.load(handle)

    return data['x'],data['y'],data['num_vertices']

num_vertices = 100
last_saved_shape_num = 0

class Train_Environment(Env):
    def __init__(self):
        self.action_space = Discrete(6)
        self.observation_space = Box(low=-5., high=5., shape=(2*num_vertices+3,), dtype=np.float32) # ????
        
        self.shape_num = 0

        state_list = []
        x, y, _= get_shape(self.shape_num)
        for i in range(num_vertices):
            state_list.append(x[i])
            state_list.append(y[i])
        state_list.append(0)
        state_list.append(0)
        state_list.append(0)
        self.state = np.array(state_list)
        shape_points_list = []
        for i in range(num_vertices):
            shape_points_list.append((x[i],y[i]))
        self.shape_poly = Polygon(shape_points_list)

        self.x_step_length = 0.01
        self.y_step_length = 0.01
        self.theta_step_length = 0.01
        self.x_total_change = 0.00001    # not set to 0 to avoid rounding errors
        self.y_total_change = 0.00001
        self.theta_total_change = 0.00001

        self.shape_poly_area = self.shape_poly.area

        self.path_found = False

        self.num_steps = 0

    def step(self, action):
        self.num_steps += 1


        x_total_change_prev = self.x_total_change
        y_total_change_prev = self.y_total_change
        theta_total_change_prev = self.theta_total_change

        if action == 0:
            self.x_total_change += self.x_step_length
        elif action == 1:
            self.x_total_change -= self.x_step_length
        elif action == 2:
            self.y_total_change += self.y_step_length
        elif action == 3:
            self.y_total_change -= self.y_step_length
        elif action == 4:
            self.theta_total_change += self.theta_step_length
        elif action == 5:
            self.theta_total_change -= self.theta_step_length


        hallway_shape, hallway_is_done_shape = transform_hallway(self.theta_total_change, self.x_total_change, self.y_total_change)
        fits, is_done = check_points(self.shape_poly, hallway_shape, hallway_is_done_shape)

        if fits == False:
            reward = -1
            self.x_total_change = x_total_change_prev
            self.y_total_change = y_total_change_prev
            self.theta_total_change = theta_total_change_prev
            done = False
        elif fits == True and is_done == False:
            reward = 0
            done = False
        else:
            reward = 10000
            self.num_steps = 0
            done = True
            
        self.state[num_vertices*2] = self.theta_total_change
        self.state[num_vertices*2 + 1] = self.x_total_change
        self.state[num_vertices*2 + 2] = self.y_total_change

        info = {}
        if self.num_steps > 100000: #arbitrary number
            draw_points_and_boundaries(self.shape_poly, hallway_is_done_shape)
            print("pathfinder couldn't find a path for self.shape_num = ",str(self.shape_num))
            # update pickle file with shapes that can't find path. Maybe just delete file and regenerate shape?
            done = True
        
        return self.state, reward, done, info

    def reset(self):
        self.num_steps = 0
        self.x_total_change = 0.00001    # not set to 0 to avoid rounding errors
        self.y_total_change = 0.00001
        self.theta_total_change = 0.00001
        
        # get next
        self.shape_num += 1
        if self.shape_num > last_saved_shape_num:
            self.shape_num = 0

        x, y, _ = get_shape(self.shape_num)
        state_list = []
        for i in range(num_vertices):
            state_list.append(x[i])
            state_list.append(y[i])
        state_list.append(self.theta_total_change)
        state_list.append(self.x_total_change)
        state_list.append(self.y_total_change)
        self.state = np.array(state_list)
        shape_points_list = []
        for i in range(num_vertices):
            shape_points_list.append((x[i],y[i]))
        self.shape_poly = Polygon(shape_points_list)

        return self.state

def Generate_Training_Shapes(num_vertices):

    x_step_length = 0.001
    y_step_length = 0.001
    
    new_area = 0
    num_vertices_to_change = 50 
    num_shapes_to_generate = 30 

    for j in range(num_shapes_to_generate):
        # generate based on this shape
        x, y = initialize_circle(num_vertices).exterior.xy
        shape_points_list = []
        for i in range(num_vertices):
            shape_points_list.append((x[i],y[i]))
        initial_shape = Polygon(shape_points_list) #initialize_circle(num_vertices)

        initial_area = initial_shape.area

        for _ in range(num_vertices_to_change):

            while new_area <= initial_area:
                x, y = initial_shape.exterior.xy

                rand_vertice = random.randint(0,num_vertices-1)
                rand_sign_x = random.randint(0,1) # 0 is negative, 1 is positive
                rand_sign_y = random.randint(0,1)

                if rand_sign_x == 0:
                    x[rand_vertice] -= x_step_length
                else:
                    x[rand_vertice] += x_step_length
                if rand_sign_y == 0:
                    y[rand_vertice] -= y_step_length
                else:
                    y[rand_vertice] += y_step_length


                shape_points_list = []
                for i in range(num_vertices):
                    shape_points_list.append((x[i],y[i]))
                new_shape = Polygon(shape_points_list)
                new_area = new_shape.area
            initial_shape = new_shape
            initial_area = new_shape.area

        save_shape(x, y, num_vertices, j+last_saved_shape_num+1)

num_shapes_to_generate = 30
Generate_Training_Shapes(num_vertices)
last_saved_shape_num+=num_shapes_to_generate
print('last_saved_shape_num = %s' % last_saved_shape_num)

env = Train_Environment()
env = Monitor(env, 'log')

model = PPO.load('model', env=env) #PPO('MlpPolicy', env, verbose=0)
eval_env = Train_Environment()
eval_env = Monitor(eval_env)

# evaluate before training
#mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=last_saved_shape_num+1)
#print(f"mean_reward before training: {mean_reward:.2f} +/- {std_reward:.2f}")

start_time = time.time()

print('Training...')
model.learn(total_timesteps=10000000)
print("done")
print("took %s minutes" % ((time.time() - start_time)/60.))


#mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=last_saved_shape_num+1)
#print(f"mean_reward after training: {mean_reward:.2f} +/- {std_reward:.2f}")


model.save("model")
print("model saved")