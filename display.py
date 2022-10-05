import pickle
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon

def get_shape(shape_num):
    with open('shape_' + str(shape_num) + '.pickle','rb') as handle:
        data = pickle.load(handle)

    return data['x'],data['y'],data['num_vertices']

for i in range(10):
    shape_num = i
    x,y, num_vertices = get_shape(shape_num)
    points_list = []
    for i in range(num_vertices):
        points_list.append((x[i],y[i]))
    print('area of shape: %s' % Polygon(points_list).area)
    x.append(x[0])
    y.append(y[0])
    plt.plot(x,y)
    plt.show()