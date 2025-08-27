from SwarmDynamicandMotion import Swarm2DMotion
from RGBImageProcess import ProcessImage
import numpy as np
import pygame
import random
import matplotlib.pyplot as plt


class ZJCSwarmEnv:
    def __init__(self, max_deltayaw, screen_size=(1280, 1280), imagesize=256, show=False):
        pygame.init()
        self.WHITE = [255,255,255]          # path 
        self.BLACK = [0,0,0]                # path
        self.GREEN = [0,255,0]              # obstacles
        self.BLUE = [0,0,255]               # swarm
        self.RED = [255,0,0]                # target

        self.show = show
        if self.show:
            self.screen_size = screen_size
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("MR Sim Env")
            self.block_size = 5
            self.x = 0
            self.frames = []

        self.MAX_DELTAYAW = max_deltayaw
        self.imagesize = imagesize

        self.maze = np.zeros((self.imagesize, self.imagesize))
        self.image = np.zeros((self.imagesize, self.imagesize, 3))
        self.process = ProcessImage()
        self.initk = 0
        self.initc = 0
        self.initb = 0

        self.robot_pos = None
        self.init_pos = None
        self.target_center = None
        self.Swarm = None

    def set_random_pos(self):
        num1 = random.randint(0,self.imagesize-1)
        num2 = random.randint(0,self.imagesize-1)
        num1 = 220.0
        num2 = 220.0
        self.robot_pos = np.array((num1, num2))
        self.init_pos = np.array((num1, num2))

    def set_random_target(self):
        num1 = random.randint(0, self.imagesize-1)
        num2 = random.randint(0, self.imagesize-1)
        num1 = 10.0
        num2 = 10.0
        self.target_center = np.array((num1, num2))

    def swarm_step(self, action, action_num, hugenoise):
        if action_num == 1:
            self.robot_pos[0], self.robot_pos[1] = self.Swarm.get_motion(1, action*self.MAX_DELTAYAW, self.robot_pos[0], self.robot_pos[1], hugenoise)
        elif action_num == 2:
            yaw, pitch = action
            self.robot_pos[0], self.robot_pos[1] = self.Swarm.get_motion(pitch*5, yaw*self.MAX_DELTAYAW, self.robot_pos[0], self.robot_pos[1], hugenoise)

        if self.show:
            self.x += 1
            self.screen.fill(self.BLACK)
            for row in range(self.imagesize):
                for col in range(self.imagesize):
                    color = self.BLACK
                    pygame.draw.rect(self.screen, color, (col*self.block_size, row*self.block_size, self.block_size, self.block_size))

            draw_pos = np.round(self.robot_pos).astype(int)
            pygame.draw.circle(self.screen, self.BLUE, (draw_pos[1]*self.block_size + self.block_size // 2, draw_pos[0]*self.block_size + self.block_size // 2), 20)
            init_pos_px = (int(self.init_pos[1] * self.block_size + self.block_size / 2), int(self.init_pos[0] * self.block_size + self.block_size / 2))
            target_center_px = (int(self.target_center[1] * self.block_size + self.block_size / 2), int(self.target_center[0] * self.block_size + self.block_size / 2))
            pygame.draw.line(self.screen, self.WHITE, init_pos_px, target_center_px, 2)
            pygame.display.flip()           
            pygame.time.Clock().tick(10)      

        distance = np.linalg.norm(self.robot_pos - self.init_pos)
        done_boundary = bool(np.any(self.robot_pos >= self.imagesize) or np.any(self.robot_pos < 0))
        relative_angle = self.process.get_angle_swarm_target(self.init_pos[0], self.init_pos[1], self.robot_pos[0], self.robot_pos[1], nor=True)
        deviation = self.process.get_deviation_swarm_path(self.robot_pos[0], self.robot_pos[1], self.initk, self.initc, self.initb) / 20

        return distance, relative_angle, deviation, done_boundary

    def reset(self, draw_maze=False):
        self.maze = np.zeros((self.imagesize, self.imagesize))
        self.image = np.zeros((self.imagesize, self.imagesize, 3))
        self.set_random_target()
        self.set_random_pos()

        if self.target_center[0] == self.init_pos[0]:
            self.initk = 1
            self.initc = 0
            self.initb = -self.init_pos[0]
        else:
            self.initk = (self.target_center[1] - self.robot_pos[1]) / (self.target_center[0] - self.robot_pos[0])
            self.initc = -1
            self.initb = self.init_pos[1] - self.initk * self.init_pos[0]

        random_a0 = random.uniform(0.4, 2.0)
        random_deltat = random.choice([0.2, 0.4, 0.6, 0.8, 1.0])
        random_a0 = 1.0
        random_deltat = 1.0
        self.Swarm = Swarm2DMotion(a0=random_a0, deltat=random_deltat)

        if draw_maze:
            plt.figure()
            plt.imshow(self.image)
            plt.show(block=False)
            plt.pause(1)

        relative_angle = self.process.get_angle_swarm_target(self.init_pos[0], self.init_pos[1], self.target_center[0], self.target_center[1], nor=True)
        deviation = 0

        return relative_angle, deviation, random_a0, random_deltat