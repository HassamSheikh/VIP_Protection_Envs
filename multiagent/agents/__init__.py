import numpy as np
from multiagent.core import Agent,Action
from functools import reduce
from math import atan2, pi, sin, cos, sqrt
import math
from collections import defaultdict
import copy

class Participant(Agent):
    """
       The participants in the bodyguard scenario (humans or robot)
    """
    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario
        self.collide = True
        self.silent = True
        self.step_size = 0.1

    def reached_goal(self):
        return self.is_collision(self.goal_a)

    def out_of_bounds(self):
        """ returns true if this agent is out of range """
        return not ((-self.scenario.env_range <= self.state.p_pos[0] <= self.scenario.env_range) and (-self.scenario.env_range <= self.state.p_pos[1] <= self.scenario.env_range))

    def is_collision(self, other):
        """ Returns whether there was a collision between this agent and another agent or landmark
        """
        dist_min = self.size + other.size
        return self.distance(other) < dist_min

    def distance(self, other):
        """ The distance between this agent and the other agent or landmark """
        return np.linalg.norm(self.state.p_pos - other.state.p_pos)

    def reset(self, low=None, high=None):
        """ Reset the states of an agent """
        low_range = low or -self.scenario.env_range
        high_range = high or self.scenario.env_range
        self.state.p_pos = np.random.uniform(low_range, high_range, self.scenario.world.dim_p)
        self.state.p_vel = np.zeros(self.scenario.world.dim_p)



class Threat:
    """Calculates threat to the target from attackers in presence of bodyguards"""

    def __init__(self, target, bodyguards, attackers, safe_distance=0.4):
        self.target = target
        self.bodyguards = bodyguards
        self.attackers = attackers
        self.safe_distance = safe_distance

    def calculate_residual_threat_at_every_step(self):
        """Calculates the residual threat to the target from the attackers at every timestep"""

        suspected_attackers = self.find_suspected_attackers()
        threat  = self.calculate_threat_from_attackers(suspected_attackers) if suspected_attackers else 0.0
        return threat

    def find_suspected_attackers(self):
        """Finds the suspected attackers"""

        suspected_attackers=[]
        for attacker in self.attackers:
            is_threat = True
            for bodyguard in self.bodyguards:
                is_threat &= self.in_line_of_sight(self.target, bodyguard, attacker)
                if not is_threat:
                    break
            if is_threat:
                suspected_attackers.append(attacker)
        return suspected_attackers

    def in_line_of_sight(self, target, bodyguard, attacker):
        """Returns whether the vip is in the line of sight """
        return not np.isclose(attacker.distance(target), (bodyguard.distance(target) + attacker.distance(bodyguard)), rtol=0.01)

    def calculate_threat_from_attackers(self, attackers):
         """Calculate threat to the target from the attackers"""

         threat_level = list(map(lambda attacker: self.chance_of_attack(attacker), attackers))
         temp_threat_level = list(map(lambda x: 1.0 - x, threat_level))
         return (1 - reduce(lambda x, y: x*y, temp_threat_level))

    def chance_of_attack(self, attacker):
        target = self.target
        distance = target.distance(attacker)
        if distance > self.safe_distance:
            return 0
        elif target.is_collision(attacker):
            return 1
        return self.attack_probability(distance)

    def attack_probability(self, distance):
        return np.exp((np.log(0.01)/self.safe_distance)*distance)

def find_nearest_suspected_crowd_members(crowd, agent, k):
    distance_between_all_crowd = np.linalg.norm(crowd-agent, axis=1)
    nearest_k_indices = distance_between_all_crowd.argsort()[:k]
    return np.take(crowd, nearest_k_indices, axis=0), nearest_k_indices
