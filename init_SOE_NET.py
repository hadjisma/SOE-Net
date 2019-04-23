###################################################
# SOE_NET network initialization
# Coded by Isma Hadji (hadjisma@cse.yorku.ca)
###################################################
"""Initialize all the SOE_Net network parameters.
e.g: orientations and basis set
"""
import numpy as np
import tensorflow as tf
import math
import configure as cfg


def initOrientations (method,speeds,num_directions):
    
    orientations=np.zeros((num_directions,3),dtype=np.float32)
    num_directions=num_directions-2
    if method is "standard":
        thetas=((2*math.pi)/num_directions)
        orientations[0]=[0, 0, 1] 
        for i in range(speeds):
            for j in range(num_directions):
                new_theta=(math.cos(thetas*float(j)), math.sin(thetas*float(j)), 1.0)
                orientations[j+1] = new_theta
            orientations[9] = [1, 1, 0] 
    elif method is "icosahedron":
        goldenRatio = (math.sqrt(5.0)+1.0)/2.0

        orientations = [[0, goldenRatio,(1.0)/goldenRatio],
                        [-(1.0)/goldenRatio, 0, goldenRatio],
                        [goldenRatio, (1.0)/goldenRatio, 0],
                        [0, -goldenRatio,(1.0)/goldenRatio],
                        [-(1.0)/goldenRatio, 0, -goldenRatio],
                        [-goldenRatio, (1.0)/goldenRatio, 0],
                        [1, 1, -1],
                        [-1, 1, 1],
                        [-1, 1, -1],
                        [-1, -1, -1]]
    else:
        print "WRONG ORIENTATIONS CHOICE !!!"
    
    return orientations

def initSeparableFilters (name, num_taps):
    
        
    sampling_rate = 0.5
    C = 0.1840
    t = np.multiply(sampling_rate, range(-num_taps,num_taps+1))
    
    f1 = -4*C*t*(-3+ 2*np.square(t))*np.exp(-np.square(t))
    f2 = t*np.exp(-np.square(t))
    f3 = -4*C*(-1+ 2*np.square(t))*np.exp(-np.square(t))
    f4 = np.exp(-np.square(t))
    f5 = -8*C*t*np.exp(-np.square(t))
    basis = np.stack((f1, f2, f3, f4, f5), axis=0)
    basis  = np.fliplr(basis)

    #basis = tf.Variable(basis, dtype=tf.float32, name)
    return basis