class Custom_log:
    def __init__(self,episode):
        self._file_logging_active = f"logging_episode_start_{episode}.txt"
    
    def get_file_name(self):
        return self._file_logging_active
    
    def write(self,msg):
        f = open(f"logs/{self._file_logging_active}","a")
        f.write(msg+"\n")
        f.close()
