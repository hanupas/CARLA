from custom_log import Custom_log
from modifiedTensorboard import ModifiedTensorBoard
from dqnAgent import DQNAgent
import gym, time
import gym_carla
import numpy as np

log_file = Custom_log()
log_file.setup_directory()
log_file.setup_file(1)

params = {
    'port': 2000,  # connection port
    'timeout': 20, #timeout for connecting to carla
    'town': 'Town03',  # which town to simulate
    'size_xy' : 200, #resolution in square form
    'sensor_camera': True, #If True then sensor is active
    'sensor_camera_segmentation': False, #If True then sensor is active
    'sensor_collision': True, #If True then sensor is active
    'sensor_obstacle': False, #If True then sensor is active
    'brake_action' : [0,1], #Value of brake
    'log_file' : log_file, #Log File path
    'fps' : 20, #Frame per second for simulation
    'step_start' : log_file.json_read_track('step'),
    'model_file' : log_file.json_read_track('model'),
    'epsilon' : log_file.json_read_track('epsilon'),
    'epsilon_decay' : 0.99999,
    'tsb': ModifiedTensorBoard,
    'framestack' : 4,
    'im_dim': 3,
    'total_step': 50_000,
    'model_save_interval': 10,
    'min_epsilon' : 0.01
}

params['obs_shape']= [params['size_xy'],params['size_xy'], params['im_dim']*params['framestack']]


if __name__ == '__main__':
    env = gym.make('carla-v0', params=params)
    agent = DQNAgent(params=params)
    agent.get_qs(np.ones((params['obs_shape'])))
    env.reset()
    current_state = env.get_framebuffer()
    epsilon = params['epsilon']
    if params['step_start'] == 0: epsilon = 1

    model_save_counter = 0

    for step in range(params['step_start'], params['step_start']+params['total_step']):
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0,len(params['brake_action']))
        current_state, done = env.step(action)
        # update tensorboard step every episode
        agent.tensorboard.step = step
        # print("step ke ",step," dari ",params['total_step'])
        
        # Decay epsilon
        if epsilon > params['min_epsilon']:
            epsilon *= params['epsilon_decay']
            epsilon = max(params['min_epsilon'], epsilon)
            params['epsilon'] = epsilon

        if(done):
            env.close()
            current_state = env.reset()
            print("lingkungan di reset pada step ",step)
            model_save_counter += 1
            if(model_save_counter >= params['model_save_interval']): 
                agent.model.save(f'model/{agent.model_name}_step{step}_time{log_file.log_time()}.h5')
                agent.tensorboard.update_stats(step=step)
                params['log_file'].json_write_track(step,agent.model_name, params['log_file'].log_time(), params['epsilon'])
                model_save_counter = 0

        time.sleep(1/params['fps'])

        
    
    env.close()
    
