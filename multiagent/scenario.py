import numpy as np
from multiagent.agents import Threat
from multiagent.agents.vip import VIP
from multiagent.agents.bodyguard import Bodyguard
from multiagent.agents.bystander import Bystander

# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()

class VIPScenario(BaseScenario):
    def __init__(self, num_bodyguards=4, num_bystanders=10, communication=True, env_range=1.0, comm_dim=4, seed=1):
        #Setting the number of agents in scenario
        self.num_bodyguards = num_bodyguards
        self.num_bystanders = num_bystanders
        self.num_agents = num_bodyguards + num_bystanders + 1
        print("Number of bodyguards", str(self.num_bodyguards))
        print("Number of bystanders", str(self.num_bystanders))

        #Setting the communication channel
        self.communication = communication
        self.comm_dim = comm_dim
        print("Is communication on? ", str(self.communication))
        if self.communication:
            print("Communication Dimensions ", str(self.comm_dim))

        #Misc env settings
        self.env_range = env_range
        self._seed = seed
        print(seed)


############################### Observation Space ##############################
    def observation(self, agent, world):
        return agent.observation()

    def done(self, agent, world):
        return self.vip_agent.reached_goal()

    def reward(self, agent, world):
        return agent.reward(world)

    def info(self, agent, world):
        info = {"residual_threat": 0.0, "total_threat": 0.0}
        if isinstance(agent, VIP):
            info["residual_threat"]=Threat(agent, self.bodyguards, self.bystanders).calculate_residual_threat_at_every_step()
            info["total_threat"]=Threat(agent, [], self.bystanders).calculate_residual_threat_at_every_step()
        return info

######################################### Utility functions #################################################
    @property
    def bodyguards(self):
        """ Returns all the bodyguards"""
        return [agent for agent in self.world.agents if isinstance(agent, Bodyguard)]

    @property
    def bystanders(self):
        """ Returns the crowd members """
        return [agent for agent in self.world.agents if isinstance(agent, Bystander)]

    @property
    def vip_agent(self):
        """ Returns the VIP"""
        return [agent for agent in self.world.agents if isinstance(agent, VIP)][0]

    @property
    def convoy_agents(self):
        """ Returns the VIP and the bodyguards"""
        return [agent for agent in self.world.agents if not isinstance(agent, Bystander)]
