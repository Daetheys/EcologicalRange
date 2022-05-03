import uuid
import os
import time

class Trainer:
    def __init__(self,config,idx=0,logger_queue=None,agent_seed=0):
        #Save config
        self.config = config

        self.agent_seed = agent_seed

        self.idx = idx

        self.logger_queue = logger_queue

    def init_env(self):
        self.env = self.config['env_class'](**self.config['env_config'])

    def init_agent(self):
        self.agent = self.config['agent_class'](self.env,self.idx,logger_queue=self.logger_queue,seed=self.agent_seed,**self.config['agent_config'])

    def init(self):
        self.init_env()
        self.init_agent()

    def dump_config(self):
        pass

    def train(self):
        self.agent.train(self.config['nb_steps'])
        
