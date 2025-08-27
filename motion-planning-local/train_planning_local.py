import numpy as np
from SwarmEnv import ZJCSwarmEnv
from DQNAgent_planning_local import DQNAgent_local
from RGBImageProcess import ProcessImage
from tensorborad_logger import TensorBoardLogger


def train_episode(env, agent, image_process, train=True):
    raw_image = env.reset()
    distance_target, vtargetlist, vdistancelist, obstcompenlist = image_process.get_vlist(raw_image)
    state_list = np.expand_dims(np.hstack((vdistancelist, vtargetlist, obstcompenlist, distance_target)), axis=0).astype(np.float32)
    state = state_list
    done = False
    total_reward = 0
    step = 0

    while not done:
        step += 1
        if train:
            action = agent.choose_action(state, vdistancelist)
        else:
            action = agent.predict_action(state)
        raw_image, done_win, done_obst = env.planning_step(action, PITCH_ANGLE)
        done = bool(done_win or done_obst or step > TIME_LIMIT)

        if not done:
            next_distance_target, vtargetlist, vdistancelist, obstcompenlist = image_process.get_vlist(raw_image)
            state_list = np.expand_dims(np.hstack((vdistancelist, vtargetlist, obstcompenlist, next_distance_target)), axis=0).astype(np.float32)
            next_state = state_list
        elif done_win:
            next_state = np.zeros((1,ACTION_NUM,4))
            next_distance_target = np.array([0])
        else:
            next_state = state
            next_distance_target = distance_target

        reward = distance_target[0].item() - next_distance_target[0].item() - np.sum(obstcompenlist)/10

        if done_win:
            reward = DONE_REWARD
            print('Reach target!')
        elif step > TIME_LIMIT:
            reward = - DONE_REWARD
            print('Reach time limit!')
        elif done_obst:
            reward = - DONE_REWARD
            print('Fail!')
        total_reward += reward

        if train:
            agent.store_transition(state, action[0], reward, next_state, done)
            agent.train()
        state = next_state
        distance_target = next_distance_target

        if done:
            break

    return total_reward, done_win

def main(retrain=False):
    env = ZJCSwarmEnv(obst=True, screen_size=(IMAGE_SIZE*10, IMAGE_SIZE*10), imagesize=IMAGE_SIZE, show=SHOWTRAIN, max_deltayaw=MAX_DELTAYAW)
    logger = TensorBoardLogger(name=SAVE_NAME)
    agent = DQNAgent_local(STATE_DIM, ACTION_NUM, logger=logger)
    image_process = ProcessImage(ACTION_NUM, IMAGE_SIZE, PITCH_ANGLE*NEXT_COE, PITCH_ANGLE*SAFETY_COE, PITCH_ANGLE*DETECTION_COE)
    done_win_list = []
    aver_reward_list = []

    if retrain:
        agent.load_model('local_model/model_for_local') 

    for i in range(MAX_EPISODE):
        total_reward, done_win = train_episode(env, agent, image_process)

        logger.log_scalar('episode reward', total_reward, i)
        logger.log_scalar('done_win', done_win, i)
        done_win_list.append(done_win)
        aver_reward_list.append(total_reward)
        aver_done_win = np.mean(done_win_list[-TRAIN_LENGTH:])
        aver_reward = np.mean(aver_reward_list[-40:])
        logger.log_scalar('aver done_win last 100', aver_done_win, i)
        logger.log_scalar('aver reward last 100', aver_reward, i)

        if aver_done_win > REWARD_TARGET and i > 100:
            agent.save_model(SAVE_NAME)
            break
    agent.save_model(SAVE_NAME)

def main_test(load_name):
    env = ZJCSwarmEnv(obst=True, screen_size=(IMAGE_SIZE*10, IMAGE_SIZE*10), imagesize=IMAGE_SIZE, show=SHOWTEST, max_deltayaw=MAX_DELTAYAW)
    agent = DQNAgent_local(STATE_DIM, ACTION_NUM)
    image_process = ProcessImage(ACTION_NUM, IMAGE_SIZE, PITCH_ANGLE*NEXT_COE, PITCH_ANGLE*SAFETY_COE, PITCH_ANGLE*DETECTION_COE)
    agent.load_model(load_name)
    reward_list = []

    for i in range(TEST_EPISODE):
        eval_reward, _ = train_episode(env, agent, image_process, train=False)
        reward_list.append(_)

    aver_reward = np.mean(reward_list[:])
    print("aver success:", aver_reward)

TIME_LIMIT = 200
ACTION_NUM = 8
MAX_DELTAYAW = 360
PITCH_ANGLE = 1
NEXT_COE = 2
SAFETY_COE = 4
DETECTION_COE = 32

#####  Adjust the IMAGE_SIZE to use different resolution environments.
IMAGE_SIZE = 64
STATE_DIM = (IMAGE_SIZE, IMAGE_SIZE, 3)

#####  training para
TRAIN_LENGTH = 100
DONE_REWARD = 10
REWARD_TARGET = 0.99
MAX_EPISODE = 1000
SHOWTRAIN = False

####   test para
TEST_EPISODE = 1000
SHOWTEST = True

SAVE_NAME = 'local_model/model_for_local'
if __name__ == '__main__':
    # main(retrain=False)
    main_test(SAVE_NAME)