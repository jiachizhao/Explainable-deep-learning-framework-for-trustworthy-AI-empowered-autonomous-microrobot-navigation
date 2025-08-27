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

    def dynamics(self, pitch, yaw, a0, noise):
        if noise:
            xnoise, ynoise = self.gen_noise()
        else:
            xnoise, ynoise = 0, 0
        x_ = (a0 * pitch * math.cos(math.radians(yaw))) * (1 + xnoise)
        y_ = (a0 * pitch * math.sin(math.radians(yaw))) * (1 + ynoise)

        return x_, y_


class Swarm2DMotion:
    def __init__(self, a0):
        self.mintime = 1
        self.a0 = a0
        self.dynamic = Swarm2DDynamic()

    def get_motion(self, pitch, yaw, xnow, ynow, hugenoise, noise):
        deltax, deltay = 0, 0
        
        for i in range(int(1/self.mintime)):
            x_, y_ = self.dynamic.dynamics(pitch, yaw, self.a0, noise)
            deltax += self.mintime * x_
            deltay += self.mintime * y_

        xnow += deltax
        ynow += deltay

        if hugenoise:
            if random.randint(1, 10) == 5:
                if random.randint(1, 2) == 1:
                    xnow += 2*pitch*self.a0
                else:
                    ynow += 2*pitch*self.a0

        xnow = np.round(xnow).astype(int)
        ynow = np.round(ynow).astype(int)

        return xnow, ynow