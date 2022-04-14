import uuid
import os

class Trainer:
    def __init__(self,config):
        self.name = config['name']
        if self.name is None:
            self.name = str(uuid.uuid4())

        seed = config['seed']

        os.makedirs(os.path.join('trainings',self.name,'logs'))

        #Logger
        logger_file_path = os.path.join('trainings',self.name,'logs')
        self.logger = config['logger_class'](logger_file_path,**config['logger_config'])
        self.logger.start()

        #Plotter
        self.plotter = config['plotter_class'](logger_file_path,**config['plotter_config'])
        self.plotter.start()

        #Env
        self.env = config['env_class'](seed=seed,**config['env_config'])

        #Agent
        self.agent = config['agent_class'](self.env,logger=self.logger,seed=seed,**config['agent_config'])

        #Save config
        self.config = config

    def dump_config(self):
        pass

    def start(self):
        print('Starting Training : ',self.name)
        self.agent.train(self.config['nb_steps'])