import gym, cv2
import sys,glob,os, time, math
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (sys.version_info.major, sys.version_info.minor,'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from gym import spaces
from gym.utils import seeding
import carla


class CarlaEnv(gym.Env):
    def __init__(self, params):
        self.log_file = params['log_file']
        self.log_file.write(f"Hai saya gym environment {params['port']}")
        self.actions = params['brake_action']
        self.action_space = spaces.Discrete(len(self.actions))
        self.img_size = (params['size_xy'], params['size_xy'])
        self.observation_space = spaces.Box(0.0, 1.0, (self.img_size[0], self.img_size[1], 1))

        client = carla.Client('localhost', params['port'])
        client.set_timeout(params['timeout'])
        self.world = client.load_world(params['town'])


        self.spawn_point = self.world.get_map().get_spawn_points()
        self.ego_bp = self.world.get_blueprint_library().filter('Firetruck')[0]
        self.npc_bp = self.world.get_blueprint_library().filter('model3')[0]
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.col_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(params['size_xy']))
        self.camera_bp.set_attribute('image_size_y', str(params['size_xy']))
        self.camera_bp.set_attribute('fov', '110')
        
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        self.done = False
        self.framebuffer = np.zeros(params['obs_shape'], 'float32')
        self.n_frames = params['framestack']
        print('Carla server connected!')
    
    def reset(self):
        self.vehicle_list = []
        self.sensor_list = []
        self.collision_hist = [] 
        self.done = False

        self.camera_img = None
        
        self.ego = self.world.spawn_actor(self.ego_bp, self.spawn_point[1])
        self.npc = self.world.spawn_actor(self.npc_bp, self.spawn_point[18])

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, carla.Transform(carla.Location(x=4,z=0.7)), attach_to=self.ego)
        self.collision_sensor = self.world.spawn_actor(self.col_bp, carla.Transform(carla.Location(x=4,z=0.7)), attach_to=self.ego)

        self.vehicle_list.append(self.ego)
        self.vehicle_list.append(self.npc)
        
        self.sensor_list.append(self.camera_sensor)
        self.sensor_list.append(self.collision_sensor)

        self.camera_sensor.listen(lambda data: self.get_camera_img(data))
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        while self.camera_img is None:
            # print("delay reset nunggu camera")
            time.sleep(0.01)

        return self.camera_img
    
    def get_camera_img(self, data):
        array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_img = array
        # print("di listen sama get camera img cuy")
        cv2.imshow("",self.camera_img)
        cv2.waitKey(1)
    
    def reset_buffer(self):
        self.obs = None
        self.framebuffer = np.zeros_like(self.framebuffer)
        return self.framebuffer

    def update_buffer(self,img):
        cropped_framebuffer = self.framebuffer[:,:,:-3]
        self.framebuffer = np.concatenate([img,cropped_framebuffer],axis=-1)

    def get_framebuffer(self):
        self.reset_buffer()
        for i in range(self.n_frames):
            self.update_buffer(self.camera_img)
            time.sleep(0.05)
        return self.framebuffer
    
    # method untuk menyimpan history collision
    def collision_data(self,event):
        self.collision_hist.append(event)

    def step(self, action):
        self.ego.apply_control(carla.VehicleControl(throttle=0.3))
        distance = lambda l: math.sqrt((l.x - self.ego.get_transform().location.x)**2 + (l.y - self.ego.get_transform().location.y)**2 + (l.z - self.ego.get_transform().location.z)**2)
        d = distance(self.vehicle_list[1].get_location())
        # self.log_file.write(f"DISTANCE = {d}")

        if len(self.collision_hist) != 0:
            self.done = True

        return self.get_framebuffer(), self.done
    
    def close(self):
        for actor in self.vehicle_list:
            actor.destroy()
        
        for sensor in self.sensor_list:
            sensor.destroy()
 
