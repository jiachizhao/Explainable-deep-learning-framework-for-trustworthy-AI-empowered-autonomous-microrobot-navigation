import pygame
import matplotlib.pyplot as plt
import numpy as np
import os
from SwarmEnv_hard_imitation import ZJCSwarmEnv
from datetime import datetime

DIRECTION_MAP = {
    pygame.K_KP1: (7),
    pygame.K_KP2: (0),
    pygame.K_KP3: (1),
    pygame.K_KP4: (6),
    pygame.K_KP6: (2),
    pygame.K_KP7: (5),
    pygame.K_KP8: (4),
    pygame.K_KP9: (3),
}

def main():
    for i in range(200):
        screen_size = (320, 320)
        imagesize = 32
        env = ZJCSwarmEnv(obst=True, screen_size=screen_size, imagesize=imagesize, show=True)
        image, finename = env.reset(draw_maze=False)

        game_data = []
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in DIRECTION_MAP:
                        action = DIRECTION_MAP[event.key]
                        game_data.append((np.copy(image), action))
                        image, done_win, done_obst = env.planning_step(action)

                        if done_win:
                            print("You reached the target!")
                            save_game_data(game_data, output_dir="D:/dataset_for_GMP", filename=finename)
                            running = False
                        elif done_obst:
                            print("You hit an obstacle!")
                            running = False

                    elif event.key == pygame.K_KP_ENTER:
                        print("Enter key pressed. Ending current loop...")
                        save_game_data(game_data, output_dir="D:/dataset_for_GMP", filename=finename)
                        running = False
            
        print("Game Over!")

def save_game_data(game_data, output_dir, filename):
    date_str = datetime.now().strftime("%Y%m%d")
    for idx, (image, key) in enumerate(game_data):
        plt.imsave(os.path.join(output_dir, f"{date_str}_{filename}_{idx:04d}_{key}.png"), image/255.0)

if __name__ == "__main__":
    main()