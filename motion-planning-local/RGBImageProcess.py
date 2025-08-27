import numpy as np
import math


class ProcessImage():
    def __init__(self, action_nums, image_size, next_dis, safe_dis, detect_dis):
        self.WHITE = [255,255,255]          # path 
        self.BLACK = [0,0,0]                # path
        self.GREEN = [0,255,0]              # obstacles
        self.BLUE = [0,0,255]               # swarm
        self.RED = [255,0,0]                # target

        self.action_nums = action_nums
        self.imagesize = image_size
        self.obst_value = -1
        self.next_distance = next_dis
        self.safety_distance = safe_dis
        self.detection_distance = detect_dis

        self.num_all_points = action_nums + 1
        self.v_list = np.zeros(self.num_all_points)
        self.vtargetlist = np.zeros((action_nums,1))
        self.directions = [
            np.array([1, 0]),   # 0°
            np.array([1, 1]),   # 45°
            np.array([0, 1]),   # 90°
            np.array([-1, 1]),  # 135°
            np.array([-1, 0]),  # 180°
            np.array([-1, -1]), # 225°
            np.array([0, -1]),  # 270°
            np.array([1, -1])   # 315°
        ]

    def get_angle_swarm_target(self, x, y, xtarget, ytarget, nor=True):
        angle = math.atan2(ytarget - y, xtarget - x)
        if nor:
            angle = math.degrees(angle)
            if angle >= 0:
                angle = angle / 360
            else:
                angle = (angle + 360) / 360

        return angle
    
    def get_V_swarm_target(self, relative_angle, consider_distance):
        if not consider_distance:
            for i in range(self.action_nums):
                angle = relative_angle - i * 0.7854
                self.vtargetlist[i] = math.cos(angle)
            self.vtargetlist += 1.0

    def get_obst_compensate(self, obstlist):
        compenlist = [
            obstlist[(i + 4) % self.action_nums] +
            obstlist[(i + 3) % self.action_nums] * 0.707 + 
            obstlist[(i + 5) % self.action_nums] * 0.707
            for i in range(self.action_nums)
            ]
        compenlist = np.array(compenlist).reshape(self.action_nums,1) 
        compenlist = np.power(compenlist,4) / (self.detection_distance * self.detection_distance)

        return abs(compenlist)

    def get_distance_swarm_obst(self, robot_pos, image):
        distances = []

        for direction in self.directions:
            distance = 0
            current_position = robot_pos.copy()
            
            while True:
                current_position += direction
                distance += 1

                if (current_position < 0).any() or (current_position >= self.imagesize).any():
                    distances.append(distance)
                    break
                
                if np.all(image[tuple(current_position.astype(int))] == self.GREEN):
                    distances.append(distance)
                    break

                if np.all(image[tuple(current_position.astype(int))] == self.RED):
                    distances.append(-1)
                    break

                if distance >= self.detection_distance:
                    distances.append(distance)
                    break

        distances = np.array(distances)
        max_distance = np.max(distances)
        distances[distances == -1] = max_distance

        safe_list = distances.reshape(self.action_nums,1) - self.safety_distance
        safe_list[safe_list > 0] = 0

        return (distances.reshape(self.action_nums,1) - self.next_distance) / self.safety_distance, self.get_obst_compensate(safe_list), distances

    def get_center(self, current_image):
        total_iswarm = 0
        total_jswarm = 0
        count_swarm = 0
        total_itarget = 0
        total_jtarget = 0
        count_target = 0

        for i in range(self.imagesize):
            for j in range(self.imagesize):
                if np.array_equal(current_image[i,j], self.BLUE):
                    total_iswarm += i
                    total_jswarm += j
                    count_swarm += 1
                if np.array_equal(current_image[i,j], self.RED):
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

    def get_vlist(self, current_image):
        center_x, center_y, xtarget, ytarget = self.get_center(current_image)
        relative_angle = self.get_angle_swarm_target(center_x, center_y, xtarget, ytarget, nor=False)
        self.get_V_swarm_target(relative_angle, consider_distance=False)
        distance_obst, obst_compensate, distances = self.get_distance_swarm_obst([center_x, center_y], current_image)
        distance_target = math.sqrt((xtarget - center_x) ** 2 + (ytarget - center_y) ** 2)
        distance_target_list = np.full((self.action_nums,1), distance_target)

        return distance_target_list, self.vtargetlist, distance_obst, obst_compensate