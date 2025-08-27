from SwarmDynamicandMotion import Swarm2DMotion
import numpy as np
import pygame
import random
import os
import matplotlib.pyplot as plt


class ZJCSwarmEnv:
    def __init__(self, obst, screen_size, imagesize, show=False, max_deltayaw=360):
        pygame.init()
        self.WHITE = [255,255,255]          # path 
        self.BLACK = [0,0,0]                # path
        self.GREEN = [0,255,0]              # obstacles
        self.BLUE = [0,0,255]               # swarm
        self.RED = [255,0,0]                # target

        self.path = 'D:/dataset/dataset_for_global_collection'
        self.image_files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        self.obst = obst
        self.show = show
        if self.show:
            self.screen_size = screen_size
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("MR Sim Env")
            self.block_size = 10

        self.MAX_DELTAYAW = max_deltayaw
        self.imagesize = imagesize

        self.maze = np.zeros((self.imagesize, self.imagesize))
        self.image = np.zeros((self.imagesize, self.imagesize, 3))

        self.robot_pos = None
        self.robot_area = None
        self.target_center = None
        self.target_area = None
        self.Swarm = None
        self.image_path = None

    def get_maze_from_image(self):
        selected_image_file = random.choice(self.image_files)
        self.image_path = os.path.join(self.path, selected_image_file)
        image = np.array(plt.imread(self.image_path))
        image = (image * 255).astype(np.uint8)
        self.target_area = []
        self.robot_area = []
        for i in range(self.imagesize):
            for j in range(self.imagesize):
                if np.array_equal(image[i,j], self.GREEN):
                    self.maze[i,j] = 1
                elif np.array_equal(image[i,j], self.RED):
                    self.target_area.append([i, j])
                elif np.array_equal(image[i,j], self.BLUE):
                    self.robot_area.append([i, j])

    def get_image_from_maze(self, done_boundary=False):
        self.image.fill(0)
        for i in range(self.imagesize):
            for j in range(self.imagesize):
                if self.maze[i, j] == 1:
                    self.image[i, j, :] = self.GREEN
                elif self.maze[i, j] == 2:
                    self.image[i, j, :] = self.RED

        if not done_boundary:
            self.image[self.robot_pos[0],self.robot_pos[1],:] = self.BLUE

    def set_random_pos(self):
        while True:
            num1 = random.randint(0,self.imagesize-1)
            num2 = random.randint(0,self.imagesize-1)
            if [num1,num2] in self.robot_area:
                return [num1, num2]

    def set_random_target(self):
        while True:
            num1 = random.randint(0, self.imagesize-1)
            num2 = random.randint(0, self.imagesize-1)
            if [num1, num2] in self.target_area:
                self.target_center = [num1, num2]
                self.target_area = []
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        x, y = num1 + i, num2 + j
                        if 0 <= x < self.imagesize and 0 <= y < self.imagesize:
                            self.target_area.append([x, y])
                            self.maze[x, y] = 2
                break

    def planning_step(self, action):
        action_angle = action * 45
        self.robot_pos[0], self.robot_pos[1] = self.Swarm.get_motion(1, action_angle, self.robot_pos[0], self.robot_pos[1], hugenoise=False, noise=False)
        self.robot_pos = np.round(self.robot_pos).astype(int)

        if self.show:
            self.screen.fill(self.BLACK)
            for row in range(self.imagesize):
                for col in range(self.imagesize):
                    color = self.BLACK if self.maze[row][col] == 0 else self.GREEN
                    pygame.draw.rect(self.screen, color, (col*self.block_size, row*self.block_size, self.block_size, self.block_size))
            for pos in self.target_area:
                pygame.draw.rect(self.screen, self.RED, (pos[1]*self.block_size, pos[0]*self.block_size, self.block_size, self.block_size))
            pygame.draw.rect(self.screen, self.BLUE, (self.robot_pos[1]*self.block_size, self.robot_pos[0]*self.block_size, self.block_size, self.block_size))
            pygame.display.flip()             
            pygame.time.Clock().tick(10)       
        
        done_win = False
        for pos in self.target_area:
            if np.all(self.robot_pos == pos):
                done_win = True
                break
        
        done_obst = False
        self.get_image_from_maze()

        if np.array_equal(self.maze[self.robot_pos[0],self.robot_pos[1]], 1):
            done_obst = True
            print(self.image_path)

        return self.image, done_win, done_obst

    def reset(self, draw_maze=False):
        self.maze = np.zeros((self.imagesize, self.imagesize))
        self.image = np.zeros((self.imagesize, self.imagesize, 3))
        if self.obst:
            self.get_maze_from_image()
        self.set_random_target()
        self.robot_pos = self.set_random_pos()
        self.get_image_from_maze()
        self.Swarm = Swarm2DMotion(a0=1.0)

        if self.show:
            self.screen.fill(self.BLACK)
            for row in range(self.imagesize):
                for col in range(self.imagesize):
                    color = self.BLACK if self.maze[row][col] == 0 else self.GREEN
                    pygame.draw.rect(self.screen, color, (col*self.block_size, row*self.block_size, self.block_size, self.block_size))
            for pos in self.target_area:
                pygame.draw.rect(self.screen, self.RED, (pos[1]*self.block_size, pos[0]*self.block_size, self.block_size, self.block_size))
            pygame.draw.rect(self.screen, self.BLUE, (self.robot_pos[1]*self.block_size, self.robot_pos[0]*self.block_size, self.block_size, self.block_size))
            pygame.display.flip()             
            pygame.time.Clock().tick(10)     

        if draw_maze:
            plt.figure()
            plt.imshow(self.image)
            plt.show(block=False)
            plt.pause(1)

        return self.image, os.path.splitext(os.path.basename(self.image_path))[0]