import numpy as np
from SwarmEnv_hard import ZJCSwarmEnv
from DQNAgent_planning_local import DQNAgent_local
from RGBImageProcess import ProcessImage
from tensorflow.keras.models import load_model

loaded_model = load_model('saved_model/my_model')
def predict_action(state, avoid_obst=None, returnaction=False):
    q_values = loaded_model.predict(state, verbose=0)
    
    if returnaction:
        avoid_obst = (avoid_obst.reshape(8) > 0).astype(int)
        q_values = q_values[0] * avoid_obst
        all_zero = np.any(q_values > 0)
        return [np.argmax(q_values), all_zero]
    else:
        return q_values

def preprocess_state(state):
    state = state / 255.0
    state = np.expand_dims(state, axis=0).astype(np.float32)

    return state

def train_episode(env, agent_local=None, image_process=None):
    raw_image = env.reset()
    distance_target, vtargetlist, vdistancelist, obstcompenlist, banlance_coe = image_process.get_vlist(raw_image)

    if agent_local:
        state_list = np.expand_dims(np.hstack((vdistancelist, vtargetlist, obstcompenlist, distance_target)), axis=0).astype(np.float32)
        state = state_list
    raw_image = preprocess_state(raw_image)
    done = False
    step = 0

    while not done:
        step += 1
        if agent_local:
            if banlance_coe <= 1:
                action = agent_local.predict_action(state)
            elif banlance_coe >= 4:
                action = predict_action(raw_image, vdistancelist, returnaction=True)
            else:
                action_global = predict_action(raw_image) ** (banlance_coe)
                action = agent_local.predict_action(state, action_global)
        else:
            action = predict_action(raw_image, vdistancelist, returnaction=True)

        next_raw_image, done_win, done_obst = env.planning_step(action, PITCH_ANGLE)
        done = bool(done_win or done_obst or step > TIME_LIMIT)

        if not done:
            next_distance_target, vtargetlist, vdistancelist, obstcompenlist, banlance_coe = image_process.get_vlist(next_raw_image)
            state_list = np.expand_dims(np.hstack((vdistancelist, vtargetlist, obstcompenlist, next_distance_target)), axis=0).astype(np.float32)
            next_state = state_list
            next_raw_image = preprocess_state(next_raw_image)
        elif done_win:
            next_state = np.zeros((1,ACTION_NUM,4))
            next_raw_image = np.zeros((1,IMAGE_SIZE,IMAGE_SIZE,3))
            next_distance_target = np.array([0])
        else:
            next_state = state
            next_raw_image = raw_image
            next_distance_target = distance_target

        if done_win:
            print('Reach target!')
        elif step > TIME_LIMIT:
            print('Reach time limit!')
        elif done_obst:
            print('Fail!')

        state = next_state
        raw_image = next_raw_image
        distance_target = next_distance_target

        if done:
            break

    return done_win

def main_test(load_name, iflocal=None):
    env = ZJCSwarmEnv(obst=True, screen_size=(IMAGE_SIZE*10, IMAGE_SIZE*10), imagesize=IMAGE_SIZE, show=SHOWTEST, max_deltayaw=MAX_DELTAYAW)
    agent_local = None
    if iflocal:
        agent_local = DQNAgent_local(STATE_DIM, ACTION_NUM)
        agent_local.load_model(load_name)
    image_process = ProcessImage(ACTION_NUM, IMAGE_SIZE, PITCH_ANGLE*NEXT_COE, PITCH_ANGLE*SAFETY_COE, PITCH_ANGLE*DETECTION_COE)
    succ_list = []

    for i in range(TEST_EPISODE):
        eval_succ = train_episode(env, agent_local, image_process)
        succ_list.append(eval_succ)
        aver_succ = np.mean(succ_list)
        print("aver success:", aver_succ)

TIME_LIMIT = 50
ACTION_NUM = 8
MAX_DELTAYAW = 360
IMAGE_SIZE = 32
PITCH_ANGLE = 1
NEXT_COE = 2
SAFETY_COE = 4
DETECTION_COE = 32
STATE_DIM = (IMAGE_SIZE, IMAGE_SIZE, 3)
TEST_EPISODE = 1000

SHOWTEST = True
if __name__ == '__main__':
    main_test('saved_model/model_for_local', iflocal=True)