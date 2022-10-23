import pickle
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
import math
import numpy as np

def get_shape(shape_num):
    with open('shape_' + str(shape_num) + '.pickle','rb') as handle:
        data = pickle.load(handle)

    return data['x'],data['y'],data['num_vertices']

def sort_vertices(shape_points_list,num_vertices): # sort vertices based on distance from each other to avoid lines cutting through middle of shape
    points_left = shape_points_list
    shape_points_list = []
    while len(points_left) != 1:
        shape_points_list.append(points_left[0])

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
    shape_points_list.append(points_left[0]) # append last point

    # update other attributes
    x = []
    y = []
    for i in range(num_vertices):
        x.append(shape_points_list[i][0])
        y.append(shape_points_list[i][1])

    return


shape_num = 0
x,y, num_vertices = get_shape(shape_num)
points_list = []
for i in range(num_vertices):
    points_list.append((x[i],y[i]))
print('area of shape: %s' % Polygon(points_list).area)
x.append(x[0])
y.append(y[0])
plt.plot(x,y)
plt.show()