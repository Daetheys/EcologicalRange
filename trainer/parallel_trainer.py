from trainer.trainer import Trainer

class ParallelTrainer(Trainer):
    def __init__(self,*args,nb_agents=2,**kwargs):
        self.nb_agents = nb_agents

    def init_env(self):
        self.envs = [self.config['env_class'](seed=self.seed,**self.config['env_config']) for i in range(len(self.nb_agents))]

    def init_agent(self):
        self.agents = [self.config['agent_class'](env,0,logger_queue=self.logger.queue,seed=self.seed,**self.config['agent_config']) for env in self.envs]
