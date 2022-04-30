import uuid
import os
import time

class Trainer:
    def __init__(self,config):
        #Save config
        self.config = config

        self.name = self.config['name']
        if self.name is None:
            self.name = str(uuid.uuid4())

        self.seed = self.config['seed']

        self.logger_file_path = os.path.join('TRAININGS',self.name,'logs')

    def create_dir(self):
        os.makedirs(os.path.join('TRAININGS',self.name,'logs'))

    def init_logger(self):
        self.logger = self.config['logger_class'](self.logger_file_path,**self.config['logger_config'])
        self.logger.init()

    def init_plotter(self):
        self.plotter = self.config['plotter_class'](self.logger_file_path,**self.config['plotter_config'])

    def init_env(self):
        self.env = self.config['env_class'](seed=self.seed,**self.config['env_config'])

    def init_agent(self):
        self.agent = self.config['agent_class'](self.env,0,logger_queue=self.logger.queue,seed=self.seed,**self.config['agent_config'])

    def init(self):
        self.create_dir()
        self.init_logger()
        self.init_plotter()
        self.init_env()
        self.init_agent()

    def start_logger(self):
        self.logger.start()

    def start_plotter(self):
        self.plotter.start()

    def dump_config(self):
        pass

    def start(self):
        print('Starting Training : ',self.name)
        self.agent.train(self.config['nb_steps'])
        print('Training Finished : ',self.name)

        #Wait for the logger to get every input and flush it
        time.sleep(1.)
        self.logger.flush()
        self.logger.stop()
        #self.plotter.stop()
