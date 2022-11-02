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
import copy

def initialize_circle(num_vertices,vertical_shift,radius = 0.2):
    assert num_vertices%2 == 0

    vertices = np.zeros(num_vertices*2) #create x (even) and y (odd)

    radius_squared = radius * radius

    x_increment = 2*radius/((num_vertices/2)-1) # ensure circle has domain [-radius*2,0]
    # upper half
    for j in range(0, int(num_vertices), 2):
        # x value
        vertices[j] = j * x_increment/2.0001

        # y value
        vertices[j + 1] = 0.99*(math.sqrt(radius_squared - pow(vertices[j] - radius, 2))) #0.99 prevents rounding errors

    # lower half
    for j in range(0, int(num_vertices), 2):
        # x value
        vertices[j + int(num_vertices)] = (2*radius-x_increment*j/2.0001)

        # y value
        vertices[j + int(num_vertices) + 1] = 0.99*(-math.sqrt(radius_squared - pow(vertices[j + int(num_vertices)] - radius, 2))) #0.99 prevents rounding errors

    for i in range(0,int(num_vertices*2),2):
        vertices[i] = vertices[i] - 1

    # put into list
    shape_list = []
    for i in range(0, int(num_vertices*2), 2):
        shape_list.append((vertices[i] - 1.5, vertices[i+1]+vertical_shift)) # -1.5 to shift to left
    
    shape = Polygon(shape_list) # polygon adds an extra vertice at the end for some reason
    return shape


radius = math.sqrt(2)/2
hallway_length = 4
# see desmos file: https://www.desmos.com/calculator/kiroz7hptu 
def transform_hallway(theta_total_change, x_total_change, y_total_change):

    # theta must be between 0 and 2pi (weird stuff happens if its not)
    if theta_total_change > 0:
        sign = 1 # positive
    else:
        sign = -1 #negative
    while theta_total_change > 2*math.pi or theta_total_change < 0:
        theta_total_change -= sign * 2 * math.pi


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

    with open('shape_' + str(shape_num) + '.pickle','wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_shape(shape_num):
    with open('shape_' + str(shape_num) + '.pickle','rb') as handle:
        data = pickle.load(handle)

    return data['x'],data['y'],data['num_vertices']

def save_params(step_length,num_vertices_to_change,time_steps):
    data = {
        'step_length': step_length,
        'num_vertices_to_change':num_vertices_to_change,
        'time_steps': time_steps
        }

    with open('params.pickle','wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_params():
    with open('params.pickle','rb') as handle:
        data = pickle.load(handle)

    return data['step_length'],data['num_vertices_to_change'],data['time_steps']


class Shape():
    def __init__(self, x, y, num_vertices):
        self.num_vertices = num_vertices
        self.x = x
        self.y = y
        self.shape_points_list = []
        for i in range(self.num_vertices):
            self.shape_points_list.append((self.x[i],self.y[i]))
        self.poly = Polygon(self.shape_points_list)
        self.state_list = []
        for i in range(self.num_vertices):
            self.state_list.append(self.x[i])
            self.state_list.append(self.y[i])
        self.state_list = np.array(self.state_list)
    def update_attributes(self): # for when x or y is changed
        self.shape_points_list = []
        for i in range(self.num_vertices):
            self.shape_points_list.append((self.x[i],self.y[i]))

        self.poly = Polygon(self.shape_points_list)
        
        self.state_list = []
        for i in range(self.num_vertices):
            self.state_list.append(self.x[i])
            self.state_list.append(self.y[i])
        self.state_list = np.array(self.state_list)
        return
    def sort_vertices(self): # sort vertices based on distance from each other to avoid lines cutting through middle of shape
        points_left = self.shape_points_list
        self.shape_points_list = []
        
        while len(points_left) != 1:
            self.shape_points_list.append(points_left[0])

            x = points_left[0][0]
            y = points_left[0][1]

            # find point closest to current point that hasn't already been used
            lowest_distance = 100 # distance of closest point so far, starting value doesn't really matter
            lowest_distance_point_index = None
            for i in range(1, len(points_left)):
                point_x = points_left[i][0]
                point_y = points_left[i][1]
                
                distance = math.sqrt((x - point_x)**2 + (y - point_y)**2)
                if distance < lowest_distance:
                    lowest_distance = distance
                    lowest_distance_point_index = i
            if lowest_distance_point_index == None:
                print(len(points_left))
            points_left[0] = points_left[lowest_distance_point_index]
            del points_left[lowest_distance_point_index]
        self.shape_points_list.append(points_left[0]) # append last point
        
        # update other attributes
        for i in range(self.num_vertices):
            self.x[i] = self.shape_points_list[i][0]
            self.y[i] = self.shape_points_list[i][1]
        self.poly = Polygon(self.shape_points_list)

        self.state_list = []
        for i in range(self.num_vertices):
            self.state_list.append(self.x[i])
            self.state_list.append(self.y[i])
        self.state_list = np.array(self.state_list)

        return

class Train_Environment(Env):
    def __init__(self, num_vertices, last_saved_shape_num):
        self.action_space = Discrete(6)
        self.observation_space = Box(low=-5., high=5., shape=(2*num_vertices+3,), dtype=np.float32) # ????

        self.last_saved_shape_num = last_saved_shape_num
        self.shape_num = -1     # initial shape will be 0 when reset is called
        self.num_vertices = new_shapes[0].num_vertices
        self.current_shape = None
        self.state = None

        self.x_step_length = 0.01
        self.y_step_length = 0.01
        self.theta_step_length = 0.01
        self.x_total_change = 0.00001    # not set to 0 to avoid rounding errors
        self.y_total_change = 0.00001
        self.theta_total_change = 0.00001

        self.num_steps = 0

        self.finished = False
        self.shapes_completed = np.zeros(last_saved_shape_num+1)

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
        fits, is_done = check_points(self.current_shape.poly, hallway_shape, hallway_is_done_shape)

        if fits == False:
            reward = -2
            self.x_total_change = x_total_change_prev
            self.y_total_change = y_total_change_prev
            self.theta_total_change = theta_total_change_prev
            done = False
        elif fits == True and is_done == False:
            reward = -1
            done = False
        else:
            if self.finished == False:
                self.shapes_completed[self.shape_num] = True
            reward = 10000
            self.num_steps = 0
            done = True
            
        self.state[self.num_vertices*2] = self.theta_total_change
        self.state[self.num_vertices*2 + 1] = self.x_total_change
        self.state[self.num_vertices*2 + 2] = self.y_total_change

        info = {}
        if self.num_steps > 100000 and self.finished == False: #arbitrary number
            self.shapes_completed[self.shape_num] = False
            done = True
        
        return self.state, reward, done, info

    def reset(self):
        self.num_steps = 0
        self.x_total_change = 0.00001    # not set to 0 to avoid rounding errors
        self.y_total_change = 0.00001
        self.theta_total_change = 0.00001
        
        # get next
        self.shape_num += 1

        if self.shape_num > self.last_saved_shape_num:
            self.finished = True
            self.shape_num = 0

        self.current_shape = new_shapes[self.shape_num]
        self.state = self.current_shape.state_list
        self.state = np.append(self.state,self.x_total_change)
        self.state = np.append(self.state,self.y_total_change)
        self.state = np.append(self.state,self.theta_total_change)

        return self.state
    def reset_shapes_completed_finished(self):
        self.finished = False
        self.shapes_completed = np.zeros(self.last_saved_shape_num+1)
        self.shape_num = -1
        return
def generate_initial_circles(num_shapes_to_generate, num_vertices, last_saved_shape_num,vertical_shift):
    if last_saved_shape_num == None:
        last_saved_shape_num = -1
    for i in range(num_shapes_to_generate):
        poly = initialize_circle(num_vertices,vertical_shift)
        x, y = poly.exterior.xy
        save_shape(x[:-1], y[:-1], num_vertices, i + last_saved_shape_num + 1)
    last_saved_shape_num += num_shapes_to_generate
    return last_saved_shape_num

def generate_new_shapes(shapes_to_change,num_vertices_to_change, num_vertices, step_length,lock_vertical):

    x_step_length = step_length
    y_step_length = step_length
    
    for shape_num, shape in enumerate(shapes_to_change):
        new_area = 0
        for _ in range(num_vertices_to_change):
            original_shape = copy.deepcopy(shape)
            initial_area = original_shape.poly.area
            while new_area <= initial_area:
                rand_vertice = random.randint(0,num_vertices-1) # changing this doesn't seem to do anything
                rand_sign_x = random.randint(0,2) # 0 is negative, 1 is positive, 2 is nothing
                if lock_vertical:
                    rand_sign_y = 2
                else:
                    rand_sign_y = random.randint(0,2)               
                

                if rand_sign_x == 0:
                    shape.x[rand_vertice] -= x_step_length
                elif rand_sign_x == 1:
                    shape.x[rand_vertice] += x_step_length
                if rand_sign_y == 0:
                    shape.y[rand_vertice] -= y_step_length
                elif rand_sign_y == 1:
                    shape.y[rand_vertice] += y_step_length
                
                shape.update_attributes()
                shape.sort_vertices()
                
                if shape.poly.area <= initial_area:
                    shape = copy.deepcopy(original_shape)
                else:
                    new_area = shape.poly.area
                
                    
        shapes_to_change[shape_num] = copy.deepcopy(shape)
    return shapes_to_change

def load_shapes(last_saved_shape_num):
    shapes = []
    for shape_num in range(last_saved_shape_num+1):
        x, y, num_vertices = get_shape(shape_num)
        shapes.append(Shape(x, y, num_vertices))
    return shapes

num_vertices = 20
last_saved_shape_num = None # None if no shapes saved yet
shapes_to_create = 1

vertical_shift = 0.25
lock_vertical = False
last_saved_shape_num = generate_initial_circles(shapes_to_create, num_vertices, last_saved_shape_num,vertical_shift)

#step_length, num_vertices_to_change, time_steps = get_params()
num_vertices_to_change = 1
step_length = .001
time_steps = 1000


start_time = time.time()
shapes = load_shapes(last_saved_shape_num)

new_shapes = copy.deepcopy(shapes)

new_shapes = generate_new_shapes(new_shapes, num_vertices_to_change, num_vertices, step_length,lock_vertical)
shapes_completed = np.zeros(last_saved_shape_num+1)

env = Train_Environment(num_vertices, last_saved_shape_num)
env = Monitor(env, 'log')
model = PPO('MlpPolicy', env, verbose=0) #PPO.load('model', env=env)
generation_rounds = 100000
log_interval = 30

#TODO: have variety of starting position of circles. Why isn't that one vertice changing???
for epoch_num in range(generation_rounds):
    env.reset_shapes_completed_finished()
    model.learn(total_timesteps=time_steps)
    
    for shape_num in range(len(env.shapes_completed)):
        if env.shapes_completed[shape_num]: # save only completed shapes
            shapes[shape_num] = copy.deepcopy(new_shapes[shape_num])

    if epoch_num % log_interval == 0:
        print('------------------Epoch %s------------------' % epoch_num)
        areas = []
        for shape in shapes:
            print(shape.poly.area)
            areas.append(shape.poly.area)
        print("max area: %s" % max(areas))
        model.save("model")

        # save shapes
        for id, shape in enumerate(shapes):
            save_shape(shape.x, shape.y, shape.num_vertices, id)
        
        # save parameters
        #save_params(step_length,num_vertices_to_change,time_steps)

    new_shapes = copy.deepcopy(shapes)
    new_shapes = generate_new_shapes(new_shapes, num_vertices_to_change, num_vertices, step_length,lock_vertical)

print("done. Took %s minutes" % ((time.time() - start_time)/60.))