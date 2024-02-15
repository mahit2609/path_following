import numpy as np
import sys 
import os

class distance:
    def __init__(self):
        self.t_incr = 0.001
        self.radius = 20  #the max deviation allowed from the path  
    #this function should give me the projection of the drone with respect to the future position and the end vector which is usually 
    #a vector in the line 
    def projection(pos,a,b):
        v1 = a - pos
        v2 = b - pos
        magnitude = np.linalg.norm(v2)
        v2 = v2/magnitude
        sp = np.dot(v1,v2)
        v2 * sp
        v2 + pos
        distance = np.linalg.norm(v1 - v2)
        return distance
    def pose_final(self ,v , x):
        exp_pos = x + v * self.t_incr
        return exp_pos
    def distance():
        print("function to find the a point in the desired lane or a vector in the desired lane")
    
    def force():
        #function decide the yaw and the curve in the function
        print("force")







            


