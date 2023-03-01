import json,os

class Custom_log:
    def __init__(self):
        pass
    
    def setup_directory(self):
        # Create models folder
        if not os.path.isdir('model'):
            os.makedirs('model')
        if not os.path.isdir('logs'):
            os.makedirs('logs')
    
    def setup_file(self,episode):
        self._file_logging_active = f"logging_episode_start_{episode}.txt"

    def get_file_name(self):
        return self._file_logging_active
    
    def write(self,msg):
        f = open(f"logs/{self._file_logging_active}","a")
        f.write(msg+"\n")
        f.close()
    
    def get_info(self, info):
        ntv = None
        # Opening JSON file
        f = open('track.json')

        # returns JSON object as
        # a dictionary
        data = json.load(f)
        ntv = data[info]

        # Closing file
        f.close()

        return ntv
    
    def json_write_track(self,episode, model, waktu, epsilon):
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
    
    def json_read_track(self, info):
        ntv = 0
        # Opening JSON file
        f = open('track.json')

        # returns JSON object as
        # a dictionary
        data = json.load(f)
        ntv = data[info]

        # Closing file
        f.close()

        return ntv
