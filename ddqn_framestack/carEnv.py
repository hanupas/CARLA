import os, sys, glob, time
import numpy as np
import cv2, math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (sys.version_info.major, sys.version_info.minor,'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

SHOW_PREVIEW = False

# Berikut kelas environment yang kita implementasikan
class CarEnv:
    def __init__(self, im_width, im_height, simulation_time, log_file):
        self.client = carla.Client("127.0.0.1",2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world("Town03")
        self.world = self.client.get_world()
        self.log_file = log_file
        self.front_camera = None
        self.n_frames = 4
        self.id_buffer = 0
        self.obs = None
        
        self.second_per_episodes = simulation_time

        self.im_width = im_width
        self.im_height = im_height
        self.obs_shape = [self.im_height, self.im_width, 3 * self.n_frames]
        self.framebuffer = np.zeros(self.obs_shape, 'float32')

        self.blueprint_library = self.world.get_blueprint_library()
        self.model3 = self.blueprint_library.filter('model3')[0]
        self.modelev = self.blueprint_library.filter('Firetruck')[0]
    
    def action_space(self):
        self.action_space = []
        self.action_space.append(("brake",0.0))
        self.action_space.append(("brake",0.1))
        return self.action_space
    
    def reset_buffer(self):
        self.obs = None
        self.id_buffer = 0
        self.framebuffer = np.zeros_like(self.framebuffer)
        return self.framebuffer

    def get_framebuffer(self):
        self.reset_buffer()
        for i in range(self.n_frames):
            self.update_buffer(self.front_camera)
            time.sleep(0.05)
        return self.framebuffer
    
    def update_buffer(self,img):
        # offset = self.obs_shape[-1]
        # axis = -1
        cropped_framebuffer = self.framebuffer[:,:,:-3]
        # self.log_file.write(f"CROPPED FRAMEBUFFER= {cropped_framebuffer}")
        self.framebuffer = np.concatenate([img,cropped_framebuffer],axis=-1)
        # self.log_file.write(f"FRAME BUFFER SHAPE DI UPDATE BUFFER= {self.framebuffer.shape}")

    # Method reset
    def reset(self):
        self.collision_hist = []
        self.obstacle_hist = []
        self.sensor_list = []
        self.vehicle_list = []
        self.episode_start = time.time()

        # Spawn kendaraan
        self.route = [0,1,18]
        self.idx = 1
        self.spawn_point = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(self.modelev, self.spawn_point[self.route[self.idx]])
        self.vehicle_list.append(self.vehicle)

        self.sensor_pos = carla.Transform(carla.Location(x=4,z=0.7))

        self.obstacle_sensor = self.world.get_blueprint_library().find('sensor.other.obstacle')
        self.obstacle_sensor =  self.world.spawn_actor(self.obstacle_sensor, self.sensor_pos, attach_to=self.vehicle)
        self.sensor_list.append(self.obstacle_sensor)
        self.obstacle_sensor.listen(lambda event: self.obstacle_data(event))

        self.colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(self.colsensor, self.sensor_pos, attach_to=self.vehicle)
        self.sensor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x',f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y',f'{self.im_height}')
        self.rgb_cam.set_attribute('fov','110')
        self.camera_sensor = self.world.spawn_actor(self.rgb_cam,self.sensor_pos, attach_to=self.vehicle)
        self.sensor_list.append(self.camera_sensor)
        self.camera_sensor.listen(lambda data:self.process_img(data))

        self.idx += 1
        self.npc = self.world.spawn_actor(self.model3, self.spawn_point[self.route[self.idx]])
        self.vehicle_list.append(self.npc)

        while self.front_camera is None:
            print("delay reset camera disini")
            time.sleep(0.01)
        
        self.reset_buffer()
        # self.update_buffer(self.front_camera)
        # self.id_buffer += 1
        # self.log_file.write(f"FRAME BUFFER ID = {self.id_buffer} = {self.framebuffer}")
        

    # method untuk menyimpan history collision
    def collision_data(self,event):
        self.collision_hist.append(event)

    # method untuk menyimpan history collision
    def obstacle_data(self, event):
        self.obstacle_hist.append(event)

    # method process image
    def process_img(self, image):
        # konversi menjadi array
        i = np.array(image.raw_data)

        # proses diatas menghasilkan flatten, kita bentuk ulang gambarnya menjadi RGBA
        # i2 = i.reshape((self.im_height, self.im_width,4))
        i2 = i.reshape((self.im_width, self.im_height,4))

        #hilangkan alpha sehingga RGBA menjadi RGB
        i3 = i2[:,:,:3]

        cv2.imshow("",i3)
        cv2.waitKey(1)

        self.front_camera = i3
    
    # method step untuk mengimplementasikan konsep reinforcement learning
    def step(self,action):
        distance = lambda l: math.sqrt((l.x - self.vehicle.get_transform().location.x)**2 + (l.y - self.vehicle.get_transform().location.y)**2 + (l.z - self.vehicle.get_transform().location.z)**2)
        d = distance(self.vehicle_list[1].get_location())
        self.log_file.write(f"DISTANCE = {d}")

        done = False
        reward = -0.1
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if (kmh > 10) : reward += 1
        
        if len(self.collision_hist) != 0:
            reward += -200
            done = True

        if(d < 15):
            if(kmh > 0) : 
                reward += -1
            else : 
                if self.episode_start + self.second_per_episodes < time.time():
                    if len(self.collision_hist) == 0:
                        reward += 100
            

        if self.episode_start + self.second_per_episodes < time.time():
            done = True
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, brake=self.action_space[action][1]))
        self.obs = self.get_framebuffer()
        return self.obs,reward,done

    def close(self):
        for sensor in self.sensor_list:
            sensor.destroy()
        for vehicle in self.vehicle_list:
            vehicle.destroy()
        print('\n Environment ditutup')


