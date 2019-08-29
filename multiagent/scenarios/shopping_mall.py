import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.agents import Threat
from multiagent.agents.bystander import Bystander
from multiagent.agents.vip import VIP
from multiagent.agents.bodyguard import Bodyguard
from multiagent.scenario import VIPScenario
import copy

class Scenario(VIPScenario):
    def __init__(self, num_bodyguards=1, num_bystanders=4, communication=False, env_range=1.0, comm_dim=4, seed=1):
        print("shopping_mall")
        super().__init__(num_bodyguards, num_bystanders, communication, env_range, comm_dim, seed)

    def make_world(self):
        """ Creates the world, the agents, the landmarks, the communication channels etc. These are for the time being all undifferentiated
        """
        world = World()
        self.world = world
        if self.communication:
            world.dim_c = self.comm_dim

        ### create the landmarks, among them the start and goal of VIP ###
        world.landmarks = self.create_landmarks(world, 12)

        ### create the agents ###
        for i in range(self.num_agents):
            if i == 0:
                agent = VIP(self)
            elif i <= self.num_bodyguards:
                agent = Bodyguard(self, self.communication, alpha=2, beta=1)
                agent.name = 'bodyguard_%d' % (i)
            else:
                agent = Bystander(self)
                if i%2==0:
                    agent = Bystander(self)
                agent.name = 'bystander_%d' % (i - self.num_bodyguards)
                agent.accel = 3.0
                agent.max_speed = 1.0
            world.agents.append(agent)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        """ Resets the world and agents. Chooses a new goal position for the VIP and
        arranges the bodyguards accordingly
        """
        # reset the position of the agents
        self.world = world
        for agent in world.agents:
            agent.reset()

        # set the initial state of the VIP
        goal, start = copy.deepcopy(world.landmarks[:2])
        self.vip_agent.state.p_pos = start.state.p_pos
        self.vip_agent.goal_a = goal

        # set the initial states of the bodyguards
        temp_angle = 360/self.num_bodyguards
        for i, agent in enumerate(self.bodyguards):
            agent_angle = (temp_angle)* np.pi / 180.
            agent.state.p_pos = world.agents[0].state.p_pos + np.array([np.cos(agent_angle), np.sin(agent_angle)])*agent.allowed_distance
            temp_angle += 360/self.num_bodyguards

        # set the position and goal of the bystander
        np.random.seed(None)
        np.random.seed(self._seed)
        bystander_p_pos = np.random.uniform(-self.env_range, self.env_range, (self.num_bystanders, world.dim_p))
        for idx, agent in enumerate(self.bystanders):
            agent.goal_a = np.random.choice(world.landmarks[2:])
            agent.state.p_pos = bystander_p_pos[idx]

    def create_landmarks(self, world, number_of_landmarks):
        world_landmarks = []
        for i in range(number_of_landmarks):
            landmark = Landmark()
            landmark.name = 'landmark %d' % i
            landmark.color = np.array([0.75,0.75,0.75])
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.collide = False
            landmark.movable = False
            world_landmarks.append(landmark)

        x = [0.0]
        y = [-self.env_range, self.env_range]
        for i, landmark in enumerate(world_landmarks[:2]):
            landmark.state.p_pos = np.array([ x[i%1], y[i//1]])

        x = [-.9, 0.9]
        y = [-.9, -0.5, 0, 0.5, 0.9]
        for i, landmark in enumerate(world_landmarks[2:]):
            landmark.state.p_pos = np.array([ x[i//5], y[i%5]])

        world_landmarks[0].color = np.array([0.15, 0.65, 0.15])
        return world_landmarks
