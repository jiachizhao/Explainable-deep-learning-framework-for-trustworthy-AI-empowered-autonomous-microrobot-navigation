import numpy as np
import math


class ProcessImage():
    def __init__(self, points_beyond_center=8):
        self.WHITE = [255,255,255]          # path 
        self.BLACK = [0,0,0]                # path
        self.GREEN = [0,255,0]              # obstacles
        self.BLUE = [0,0,255]               # swarm
        self.RED = [255,0,0]                # target

        self.imagesize = 128
        self.obst_value = -100
        self.num_all_points = points_beyond_center+1
        self.v_list = np.zeros(self.num_all_points)

    def get_angle_swarm_target(self, x, y, xtarget, ytarget, nor=True):
        angle = math.atan2(ytarget - y, xtarget - x)
        angle = math.degrees(angle)
        if nor:
            if angle >= 0:
                angle = angle / 360
            else:
                angle = (angle + 360) / 360

        return angle
    
    def get_deviation_swarm_path(self, x, y, k, c, b):
        return abs(k * x + c * y + b) / math.sqrt(k*k + c*c)

    def get_center(self, current_image):
        total_iswarm = 0
        total_jswarm = 0
        count_swarm = 0
        total_itarget = 0
        total_jtarget = 0
        count_target = 0

        for i in range(self.imagesize):
            for j in range(self.imagesize):
                if np.array_equal(current_image[self.imagesize-j-1,i], self.BLUE):
                    total_iswarm += i
                    total_jswarm += j
                    count_swarm += 1
                if np.array_equal(current_image[self.imagesize-j-1,i], self.RED):
                    total_itarget += i
                    total_jtarget += j
                    count_target += 1                

        if count_swarm > 0 and count_target > 0:
            swarm_x = round(total_iswarm / count_swarm)
            swarm_y = round(total_jswarm / count_swarm)
            target_x = round(total_itarget / count_target)
            target_y = round(total_jtarget / count_target)

            return swarm_x, swarm_y, target_x, target_y

        else:
            return f"swarm or target does not exist"

    def get_vlist(self, current_image, linelist, nor=True, from_torch=False):
        if from_torch:
            current_image = current_image.detach().numpy()
            current_image = ((current_image + 1) * 127.5).astype(np.uint8)
            current_image = np.transpose(current_image, (1, 2, 0))

        center_x, center_y, xtarget, ytarget = self.get_center(current_image)
        relative_angle = self.get_angle_swarm_target(center_x, center_y, xtarget, ytarget, nor=nor)
        deviation = self.get_deviation_swarm_path(center_x, center_y, linelist[0], linelist[1], linelist[2]) / 10

        all_points = self.get_allxy.get_all_radisuxy(center_x, center_y)
        if np.any(np.all(self.GREEN == current_image, axis=-1)):
            for i in range(self.num_all_points):
                x = round(all_points[i, 0])
                y = round(all_points[i, 1])
                if x < 0 or y < 0 or x >= self.imagesize or y >= self.imagesize:  
                    self.v_list[i] = 2 * self.obst_value
                elif np.array_equal(current_image[self.imagesize-y-1,x], self.GREEN):
                    self.v_list[i] = self.get_v(x, y, xtarget, ytarget) + self.obst_value
                else:
                    self.v_list[i] = self.get_v(x, y, xtarget, ytarget)
            
        else:
            for i in range(self.num_all_points):
                x = round(all_points[i, 0])
                y = round(all_points[i, 1])
                if x < 0 or y < 0 or x >= self.imagesize or y >= self.imagesize:
                    self.v_list[i] = 2 * self.obst_value
                else:
                    self.v_list[i] = self.get_v(x, y, xtarget, ytarget)
        
        self.v_list[1:] -= self.v_list[0]
        if nor:
            self.v_list[1:] *= 50
            
        return self.v_list, relative_angle, deviation