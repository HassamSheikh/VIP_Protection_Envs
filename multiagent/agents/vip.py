import numpy as np
from multiagent.core import Agent,Action
from functools import reduce
from math import atan2, pi, sin, cos, sqrt
import math
from collections import defaultdict
import copy
from . import Participant

class VIP(Participant):
    """ The VIP is moving from a start location to another
    """
    def __init__(self, scenario):
        super().__init__(scenario)
        self.name = "VIP"
        self.color = np.array([0.5, 0.25, 0.25]) # brown
        self.state.p_pos = np.random.uniform(-scenario.env_range, scenario.env_range, scenario.world.dim_p)
        self.state.p_vel = np.zeros(scenario.world.dim_p)
        self.max_speed = 1.0
        # comment out this to experiment with a manual VIP
        self.action_callback = self.theaction

    def theaction(self, agent, world):
        """ Explicitly programmed VIP behavior """
        retval = Action()
        if self.near_bystander(agent, world) or not self.goal_a:
            retval.u = np.zeros(world.dim_p)
            return retval
        relative_position = (agent.goal_a.state.p_pos - agent.state.p_pos)
        retval.u = (relative_position/np.linalg.norm(relative_position)) * self.step_size
        return retval

    def near_bystander(self, agent, world):
        bystander_p_pos = np.asarray([bystander.state.p_pos for bystander in self.scenario.bystanders])
        distance_between_all_bystanders = np.linalg.norm(bystander_p_pos-agent.state.p_pos, axis=1)
        return np.any(0.3 > distance_between_all_bystanders)

    def reward(self, world):
        """ The reward of the VIP - used when training the VIP behavior """
        residual_threat = Threat(self, self.scenario.bodyguards, self.scenario.bystanders)
        return -self.distance(self.goal_a) -2*residual_threat.calculate_residual_threat_at_every_step()

    def observation(self):
        """returns the observation of the bodyguard"""
        crowd_p_pos = np.asarray([bystander.state.p_pos for bystander in self.scenario.bystanders])
        nearest_crowd, crowd_idx = find_nearest_suspected_crowd_members(crowd_p_pos, self.state.p_pos, len(crowd_p_pos))
        other_vel = []
        for idx in crowd_idx:
            other_vel.append(self.scenario.bystanders[idx].state.p_vel)
        return np.concatenate([self.state.p_vel] + [(self.goal_a.state.p_pos-self.state.p_pos)] + [(nearest_crowd - self.state.p_pos).flatten()] + other_vel)
