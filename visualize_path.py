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
import pygame
import os

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
        self.num_vertices = shape_to_vizualize[0].num_vertices
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

        self.shapes_completed = [] # shapes that fit through hallway

        self.shape_state_history = [] # only includes theta,x,y becuase vertices are constant

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
            reward = -1
            self.x_total_change = x_total_change_prev
            self.y_total_change = y_total_change_prev
            self.theta_total_change = theta_total_change_prev
            done = False
        elif fits == True and is_done == False:
            reward = 0
            done = False
        else:
            if not self.finished:
                self.shapes_completed.append(self.current_shape)
            reward = 10000
            self.num_steps = 0
            done = True
            
        self.state[self.num_vertices*2] = self.theta_total_change
        self.state[self.num_vertices*2 + 1] = self.x_total_change
        self.state[self.num_vertices*2 + 2] = self.y_total_change

        if not self.finished:
            self.shape_state_history.append(hallway_shape)

        info = {}
        if self.num_steps > 100000: #arbitrary number
            draw_points_and_boundaries(self.current_shape.poly, hallway_is_done_shape)
            print("self.theta_total_change %s " % self.theta_total_change)
            print("self.x_total_change %s " % self.x_total_change)
            print("self.y_total_change %s " % self.y_total_change)
            print("pathfinder couldn't find a path for self.shape_num = ",str(self.shape_num))
            self.shapes_completed.append(None)
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

        self.current_shape = shape_to_vizualize[self.shape_num]
        self.state = self.current_shape.state_list
        self.state = np.append(self.state,self.x_total_change)
        self.state = np.append(self.state,self.y_total_change)
        self.state = np.append(self.state,self.theta_total_change)

        return self.state

def load_shapes(last_saved_shape_num):
    shapes = []
    for shape_num in range(last_saved_shape_num+1):
        x, y, num_vertices = get_shape(shape_num)
        shapes.append(Shape(x, y, num_vertices))
    return shapes

shape_to_vizualize_num = 0
x, y, num_vertices = get_shape(shape_to_vizualize_num)
shape_to_vizualize = [Shape(x, y, num_vertices)]
env = Train_Environment(num_vertices, 0)
env = Monitor(env, 'log')
model = PPO.load('model', env=env) # PPO('MlpPolicy', env, verbose=0)
time_steps = 200000
model.learn(total_timesteps=time_steps)

if env.finished:
    pygame.init()
    width = 1000
    height = 1000
    scale = width/4
    shift = width/2 + 150
    total_time = 10 # seconds
    time_per_state = total_time/len(env.shape_state_history)
    screen = pygame.display.set_mode((width,height))
    white = '0xffffff'
    black = '0x000000'
    print('visualizing path of shape %s' % shape_to_vizualize_num)
    for i in range(len(env.shape_state_history)):
        screen.fill(white)
        # draw lines for shape
        for l in range(num_vertices-1):
            start_point = [shift + scale*shape_to_vizualize[0].x[l],shift + scale*shape_to_vizualize[0].y[l]]
            end_point = [shift + scale*shape_to_vizualize[0].x[l+1],shift + scale*shape_to_vizualize[0].y[l+1]]
            pygame.draw.line(screen,pygame.Color(black),start_point,end_point)
        # line from last point to first point
        start_point = [shift + scale*shape_to_vizualize[0].x[num_vertices-1],shift + scale*shape_to_vizualize[0].y[num_vertices-1]]
        end_point = [shift + scale*shape_to_vizualize[0].x[0],shift + scale*shape_to_vizualize[0].y[0]]
        pygame.draw.line(screen,pygame.Color(black),start_point,end_point)

        # hallway lines
        x,y = env.shape_state_history[i].exterior.xy
        left_upper_point = [shift + scale*x[0],shift + scale*y[0]]
        left_lower_point = [shift + scale*x[1],shift + scale*y[1]]
        middle_middle = [shift + scale*x[2],shift + scale*y[2]]
        middle_lower = [shift + scale*x[3],shift + scale*y[3]]
        right_lower = [shift + scale*x[4],shift + scale*y[4]]
        right_upper = [shift + scale*x[5],shift + scale*y[5]]
        pygame.draw.line(screen,pygame.Color(black),left_upper_point,left_lower_point)
        pygame.draw.line(screen,pygame.Color(black),left_lower_point,middle_middle)
        pygame.draw.line(screen,pygame.Color(black),middle_middle,middle_lower)
        pygame.draw.line(screen,pygame.Color(black),middle_lower,right_lower)
        pygame.draw.line(screen,pygame.Color(black),right_lower,right_upper)
        pygame.draw.line(screen,pygame.Color(black),right_upper,left_upper_point)
        # sleep
        time.sleep(0.1)
        pygame.display.flip()

else:
    print('env not finished, increase time_steps')

    
os.sys.exit()
