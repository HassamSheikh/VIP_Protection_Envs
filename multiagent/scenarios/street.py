import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.agents import VIP, Bodyguard, StreetBystander
from multiagent.scenario import VIPScenario
import copy

class Scenario(VIPScenario):
    def __init__(self, num_bodyguards=4, num_bystanders=10, communication=True, env_range=1.0, comm_dim=4, seed=1):
        super().__init__(num_bodyguards, num_bystanders, communication, env_range, comm_dim, seed)

    def make_world(self):
        """ Creates the world, the agents, the landmarks, the communication channels etc. These are for the time being all undifferentiated
        """
        world = World()
        self.world = world
        if self.communication:
            world.dim_c = self.comm_dim

        ### create the landmarks, among them the start and goal of VIP ###
        world.landmarks = self.create_landmarks(world, 22)

        ### create the agents ###
        for i in range(self.num_agents):
            if i == 0:
                agent = VIP(self)
            elif i <= self.num_bodyguards:
                agent = Bodyguard(self, self.communication, alpha=2.5, beta=2)
                agent.name = 'bodyguard %d' % (i)
            else:
                agent = StreetBystander(self)
                agent.name = 'bystander %d' % (i - self.num_bodyguards)
                agent.accel = 3.0
                agent.max_speed = 1.0
            world.agents.append(agent)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        """ Resets the world and agents. Chooses a new goal position for the VIP and
        arranges the bodyguards accordingly
        """
        """ Resets the world and agents. Chooses a new goal position for the VIP and
        arranges the bodyguards accordingly
        """
        self.world = world
        for agent in world.agents:
            agent.reset()

        # set the initial state of the VIP
        goal, start = copy.deepcopy(world.landmarks[:2])
        self.vip_agent.state.p_pos = start.state.p_pos + start.size
        self.vip_agent.goal_a = goal

        # set the initial states of the bodyguards
        temp_angle = 360/self.num_bodyguards
        for i, agent in enumerate(self.bodyguards):
            agent_angle = (temp_angle)* np.pi / 180.
            agent.state.p_pos = world.agents[0].state.p_pos + np.array([np.cos(agent_angle), np.sin(agent_angle)])*agent.allowed_distance
            temp_angle += 360/self.num_bodyguards

        # set position of the bystanders behind the landmarks
        np.random.seed(seed=None)
        seed=np.random.randint(1, 10)
        np.random.seed(seed)
        bystander_theta = np.random.uniform(-np.pi, np.pi, self.num_bystanders)

        np.random.seed(seed=None)
        seed=np.random.randint(1, 10)
        np.random.seed(seed)
        bystander_noise = np.random.rand(self.num_bystanders)

        x = np.array([-0.6, .6])
        y = np.arange(-.9, 1.0, 0.4)
        bystander_p_pos=np.transpose([np.repeat(x, len(y)), np.tile(y, len(x))])
        for i, agent in enumerate(self.bystanders):
            agent.state.p_pos = bystander_p_pos[i]
            agent.theta =  bystander_theta[i]
            agent.noise = bystander_noise[i]

        # selecting the attacker from bystanders
        attacker = np.random.choice(self.bystanders)
        attacker.goal_a = self.vip_agent

    def create_landmarks(self, world, number_of_landmarks):
        world_landmarks = []
        for i in range(number_of_landmarks):
            landmark = Landmark()
            landmark.name = 'landmark %d' % i
            landmark.color = np.array([0.75,0.75,0.75])
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.050
            world_landmarks.append(landmark)

        x = np.array([0])
        y = np.array([-0.9, 0.9])
        landmark_p_pos = np.transpose([np.repeat(x, len(y)), np.tile(y, len(x))])
        for i, landmark in enumerate(world_landmarks[:2]):
            landmark.state.p_pos = landmark_p_pos[i]

        x =  np.array([-0.9, 0.9])
        y =  np.arange(-1, 1.5, 0.25)
        landmark_p_pos = np.transpose([np.repeat(x, len(y)), np.tile(y, len(x))])
        for i, landmark in enumerate(world_landmarks[2:]):
            landmark.state.p_pos = landmark_p_pos[i]

        world_landmarks[0].color = np.array([0.15, 0.65, 0.15])
        return world_landmarks
