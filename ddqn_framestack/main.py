import os, time, json
from tqdm import tqdm
from carEnv import CarEnv
from dqnAgent import DQNAgent
from custom_log import Custom_log
import numpy as np

IM_WIDTH = 120 
IM_HEIGHT = 120
epsilon = 0
EPISODES = 1000
EPISODE_START = 0
SECOND_PER_EPISODE = 20
AGGREGATE_STATS_EVERY = 10
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.999

def json_write(episode, model, waktu, epsilon):
    # Opening JSON file
    f = open('track.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    data['episode'] = episode
    if(model != "") : data['model'] = model
    data['waktu'] = waktu
    data['epsilon'] = epsilon

    with open("track.json", "w") as jsonFile:
        json.dump(data, jsonFile)
    
    # Closing file
    f.close()

def get_episode():
    ntv = 0
    # Opening JSON file
    f = open('track.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    ntv = data['episode']

    # Closing file
    f.close()

    return ntv

def get_model():
    ntv = 0
    # Opening JSON file
    f = open('track.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    ntv = data['model']

    # Closing file
    f.close()

    return ntv

def get_epsilon():
    ntv = 0
    # Opening JSON file
    f = open('track.json')

    # a dictionary
    data = json.load(f)
    ntv = data['epsilon']

    # Closing file
    f.close()

    return ntv

if __name__ == '__main__':
    try:
        # Create models folder
        if not os.path.isdir('modelhandoko'):
            os.makedirs('modelhandoko')
        if not os.path.isdir('logs'):
            os.makedirs('logs')

        # For stats
        EPISODE_START = get_episode()
        MODEL_PATH = get_model()
        epsilon = get_epsilon()
        if (EPISODE_START == 0) : epsilon = 1
        mylog = Custom_log(EPISODE_START)
        env = CarEnv(IM_WIDTH, IM_HEIGHT,SECOND_PER_EPISODE, mylog)
        action_space_size = len(env.action_space())
        observation_space_dim = env.reset_buffer().shape
        mylog.write(f"observation space dim = {observation_space_dim}")
        agent = DQNAgent(action_space_size,observation_space_dim,EPISODE_START,MODEL_PATH,mylog)
        mylog.write(f"============= START LOGGING EPISODE {EPISODE_START}========================")
        mylog.write(f"IM WIDTH = {IM_WIDTH} \n IM HEIGHT = {IM_HEIGHT}\n ACTION SPACE SIZE = {action_space_size} \n MODEL PATH = {MODEL_PATH}")
        

        # Initialize predictions - first prediction takes longer as of initialization that has to be done
        # It's better to do a first prediction then before we start iterating over episode steps
        agent.get_qs(np.ones((IM_WIDTH, IM_HEIGHT, 3*4)))
        best_average = -999

        # For stats
        ep_rewards = [0]
        for episode in tqdm(range(EPISODE_START, EPISODE_START+EPISODES), ascii=True, unit="episode"):
            done = False
            episode_reward = 0
            mylog.write(f"==================  EPISODE {episode}======================================")

            # update tensorboard step every episode
            agent.tensorboard.step = episode

            # SIMULATION STEP
            env.reset()
            current_state = env.get_framebuffer()
            while True:
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0,action_space_size)
                new_state,reward, done = env.step(action)
                # mylog.write(f"New State shape di main = {new_state.shape}")
                episode_reward += reward

                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                current_state = new_state

                if done:break
                # FPS
                time.sleep(0.05)
            env.close()
            # Save Model, but only when min reward is greater or equal a set value
            now = time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(int(time.time())))
            json_write(episode,"",now, epsilon)
            print(f"episode reward = {episode_reward}")

            # Append episode reward to a list and log stats(every given number of episode)
            ep_rewards.append(episode_reward)

            if not episode % AGGREGATE_STATS_EVERY or episode==1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg = average_reward, reward_min = min_reward, reward_max = max_reward, epsilon=epsilon, episode=episode)

                print(f"episode {episode} || average reward ={average_reward}")
                agent.train()
                if average_reward >= best_average:
                    real_episode = EPISODE_START + episode
                    agent.model.save(f'modelhandoko/{agent.model_name}_ep{real_episode}_avg{average_reward:_>7.2f}_time{now}.h5')
                    best_average = average_reward
                    json_write(episode,f"{agent.model_name}_ep{real_episode}_avg{average_reward:_>7.2f}_time{now}.h5",now,epsilon)
                    print("best reward average ="+str(best_average)+" at episode "+str(episode))
            
            #Decay Epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

    except KeyboardInterrupt:
        env.close()
        print('\nCancelled by user. Bye!')

