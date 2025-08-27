import math
import numpy as np
import random


class Swarm2DDynamic:
    def __init__(self, maxnoise=0.4): 
        self.maxnoise = maxnoise
    
    def gen_noise(self, fixrandomseed=False, randomseed=42):
        if fixrandomseed:
            random.seed(randomseed)
        xnoise = self.maxnoise * random.uniform(-1,1)
        ynoise = self.maxnoise * random.uniform(-1,1)

        return xnoise, ynoise

    def dynamics(self, pitch, yaw, a0):
        xnoise, ynoise = self.gen_noise()
        x_ = (a0 * pitch * math.cos(math.radians(yaw))) * (1 + xnoise)
        y_ = (a0 * pitch * math.sin(math.radians(yaw))) * (1 + ynoise)

        return x_, y_


class Swarm2DMotion:
    def __init__(self, a0, deltat):
        self.mintime = 1
        self.a0 = a0
        self.deltat = deltat
        self.onesteplength = self.a0 * self.deltat

        self.dynamic = Swarm2DDynamic()

    def get_motion(self, pitch, yaw, xnow, ynow, hugenoise):
        deltax, deltay = 0, 0
        
        for i in range(int(1/self.mintime)):
            x_, y_ = self.dynamic.dynamics(pitch, yaw, self.onesteplength)
            deltax += self.mintime * x_
            deltay += self.mintime * y_

        xnow += deltax
        ynow += deltay

        if hugenoise:
            if random.randint(1, 10) == 5:
                if random.randint(1, 2) == 1:
                    xnow += 2*pitch*self.onesteplength
                else:
                    ynow += 2*pitch*self.onesteplength

        return xnow, ynow