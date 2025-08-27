import numpy as np
import math
from SwarmEnv import ZJCSwarmEnv
from TD3Agent import TD3Agent
from tensorborad_logger import TensorBoardLogger


def cal_diff_angle(init_angle, rela_angle):
    diff_angle = init_angle - rela_angle
    if diff_angle > 0.5:
        diff_angle = diff_angle - 1
    elif diff_angle < -0.5:
        diff_angle = 1 + diff_angle

    return diff_angle

def train_episode(env, agent, train=True):
    init_relative_angle, deviation, a0, deltat = env.reset()
    state = np.array([0, 0, a0, deltat])
    done = False
    total_reward = 0
    step = 0
    distance = 0
    diff_record = [0]

    while not done:
        step += 1
        if train:
            action = agent.choose_action(state)
        else:
            action = agent.predict_action(state)
        action_env = action + init_relative_angle

        next_distance, relative_angle, deviation, done_boundary = env.swarm_step(action_env, ACTION_NUM, HUGENOISE)
        done_win = step > (TIME_LIMIT / deltat)
        done = bool(done_boundary or done_win)

        diff_relative_angle = cal_diff_angle(init_relative_angle, relative_angle)
        next_distance = next_distance * math.cos(math.radians(diff_relative_angle*360))
        if diff_relative_angle < 0:
            deviation *= -1
        next_state = np.array([deviation, diff_relative_angle, a0, deltat])
        diff_record.append(10*deviation)

        reward = 0.5*(next_distance-distance) - 60*deviation**2 - 2*abs(diff_relative_angle)

        if done_win:
            next_state = np.zeros(STATE_DIM)
            diff_record.append(0)
            print('Reach time limit !')
        elif done_boundary:
            diff_record.append(10*deviation)
            reward = - DONE_REWARD
            print('Fail !')

        total_reward += reward

        if train:
            agent.remember(state, action, reward, next_state, done)
            agent.train()
        state = next_state
        distance = next_distance

        if done:
            break

    return total_reward, diff_record

def main(retrain=False):
    env = ZJCSwarmEnv(show=SHOWTRAIN, max_deltayaw=MAX_DELTAYAW)
    logger = TensorBoardLogger(name=SAVE_NAME)
    agent = TD3Agent(input_dims=STATE_DIM, batch_size=BATCH_SIZE, n_actions=ACTION_NUM)
    reward_list = []
    if retrain:
        agent.load_models(checkpoint_dir='motionTD3')  

    for i in range(MAX_EPISODE):
        total_reward, diff = train_episode(env, agent)

        logger.log_scalar('episode reward', total_reward, i)
        reward_list.append(total_reward)
        aver_reward = np.mean(reward_list[-TRAIN_LENGTH:])
        logger.log_scalar('aver reward last 50', aver_reward, i)
        print("episode and reward:", i, total_reward, aver_reward)

        if aver_reward > REWARD_TARGET and i > 40:
            agent.save_models(checkpoint_dir=SAVE_NAME)
            break
    agent.save_models(checkpoint_dir=SAVE_NAME)

def main_test(load_name):
    env = ZJCSwarmEnv(show=SHOWTEST, max_deltayaw=MAX_DELTAYAW)
    agent = TD3Agent(input_dims=STATE_DIM, batch_size=BATCH_SIZE, n_actions=ACTION_NUM)
    agent.load_models(checkpoint_dir=load_name)
    reward_list = []

    for i in range(TEST_EPISODE):
        eval_reward, diff = train_episode(env, agent, train=False)
        reward_list.append(eval_reward)

    aver_reward = np.mean(reward_list[-TEST_EPISODE:])
    print(load_name + "aver reward:", aver_reward)

    return aver_reward

TIME_LIMIT = 100
ACTION_NUM = 1
MAX_DELTAYAW = 360
STATE_DIM = 4
BATCH_SIZE = 128
HUGENOISE = True

##### train para
TRAIN_LENGTH = 50
DONE_REWARD = 100
SHOWTRAIN = False
REWARD_TARGET = 100
MAX_EPISODE = 1000

##### test para
TEST_EPISODE = 100
SHOWTEST = True

SAVE_NAME = 'motionTD3'
if __name__ == '__main__':
    # main(retrain=False)
    main_test(SAVE_NAME)