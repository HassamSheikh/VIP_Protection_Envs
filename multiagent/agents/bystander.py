import numpy as np
from . import *

class Bystander(Participant):
    """ A bystander (crowd participant) in the bodyguard environment, performing a movement that involves visiting random landmarks. If the bystander is near a bodyguard, it stops...
    """
    def __init__(self, scenario):
        super().__init__(scenario)
        self.action_callback = self.theaction
        self.color = np.array([0.8, 0.0, 0.0]) # red
        self.state.p_pos = np.random.uniform(-1,+1, scenario.world.dim_p)
        self.state.p_vel = np.zeros(scenario.world.dim_p)
        self.goal_a = None
        self.wait_count = 0

    def reset(self):
        super(Bystander, self).reset()
        self.goal_a=None

    def theaction(self, agent, world):
        """ The behavior of the bystanders. Implemented as callback function
        """
        # If the agent finds itself out of range, jump to a random new location
        if self.out_of_bounds():
            self.reset()
        bystander_action = Action()
        # The bystanders freeze if they are near a bodyguard or have no goal
        if self.near_bodyguard(agent, world) or not self.goal_a:
            bystander_action.u = np.zeros(world.dim_p)
            self.wait_count += 1
            if self.wait_count > 50:
                agent.goal_a = self.nearest_landmark(world)
                relative_position = (agent.goal_a.state.p_pos - agent.state.p_pos)
                bystander_action.u = (relative_position/np.linalg.norm(relative_position))
                self.wait_count = 0
            return bystander_action
        # If the agent reached its goal, picks a new goal randomly from the landmarks
        if self.reached_goal():
            agent.goal_a = np.random.choice(world.landmarks)
        # otherwise, move towards the landmark
        relative_position = (agent.goal_a.state.p_pos - agent.state.p_pos)
        bystander_action.u = (relative_position/np.linalg.norm(relative_position)) * self.step_size
        return bystander_action

    def near_bodyguard(self, agent, world):
        bodyguard_p_pos = np.asarray([bodyguard.state.p_pos for bodyguard in self.scenario.bodyguards])
        distance_between_all_bodyguards = np.linalg.norm(bodyguard_p_pos-agent.state.p_pos, axis=1)
        return np.any(0.3 > distance_between_all_bodyguards)

    def nearest_landmark(self, world):
        landmark_p_pos = np.array([landmark.state.p_pos for landmark in world.landmarks])
        idx = np.linalg.norm(landmark_p_pos-self.state.p_pos, axis=1).argsort()[0]
        return world.landmarks[idx]

class StreetBystander(Bystander):
    """ A bystander (crowd participant) in the bodyguard environment, performing Vicsek Particle Motion. If the bystander is near a bodyguard, it stops...
    """
    def __init__(self, scenario):
        super().__init__(scenario)
        self.action_callback = self.theaction
        self.theta = np.random.uniform(-np.pi,np.pi)
        self.noise = np.random.rand()

    def reset(self):
        """ Reset the states of an agent """
        self.state.p_vel = np.random.uniform(-.5, .5, self.scenario.world.dim_p)
        self.theta=np.random.uniform(-np.pi,np.pi)

    def theaction(self, agent, world):
        """ The behavior of the bystanders. Implemented as callback function
        """
        #print("bystander action")
        # If the agent finds itself out of range, jump to a random new location
        bystander_action = Action()
        #The bystanders freeze if they are near a bodyguard
        if self.near_bodyguard(agent, world) or self.out_of_bounds():
            bystander_action.u = np.array([-0.2, -0.2])
            return bystander_action
        # otherwise, move towards the landmark
        relative_position= (self.vicsek_step() - agent.state.p_pos)
        bystander_action.u = (relative_position/np.linalg.norm(relative_position))
        return bystander_action

    def near_bodyguard(self, agent, world):
        bodyguard_p_pos = np.asarray([bodyguard.state.p_pos for bodyguard in self.scenario.bodyguards])
        distance_between_all_bodyguards = np.linalg.norm(bodyguard_p_pos-agent.state.p_pos, axis=1)
        return np.any(0.1 > distance_between_all_bodyguards)

    def vicsek_step(self):
        noise_increments = (self.noise - 0.5)
        bystander_p_pos = np.asarray([bystander.state.p_pos for bystander in self.scenario.bystanders])
        distance_between_all_crowd = np.linalg.norm(bystander_p_pos-self.state.p_pos, axis=1)
        np.nan_to_num(distance_between_all_crowd, False)
        near_range_bystanders = np.where((distance_between_all_crowd > 0) & (distance_between_all_crowd <=1.5))[0].tolist()
        near_angles = [self.scenario.bystanders[idx].theta for idx in near_range_bystanders]
        near_angles = np.array(near_angles)
        mean_directions = np.arctan2(np.mean(np.sin(near_angles)), np.mean(np.cos(near_angles)))
        self.theta =  mean_directions + noise_increments
        vel = np.multiply([np.cos(self.theta), np.sin(self.theta)], self.state.p_vel)
        position = self.state.p_pos + (vel * 0.15)
        if not ((-self.scenario.env_range <= position[0] <= self.scenario.env_range) and (-self.scenario.env_range <= position[1] <= self.scenario.env_range)):
            return copy.deepcopy(self.state.p_pos + .1)
        return np.clip(position, -1, 1)


class HostileBystander(Bystander):
    """A Hostile Bystander"""
    def __init__(self, scenario):
        super().__init__(scenario)
        self.action_callback = None
        #self.color = np.array([0.8, 0.0, 1.1])
    def observation(self):
        """returns the observation of a hostile bystander"""
        other_pos = []
        other_vel = []
        for other in self.scenario.world.agents:
            if other is self: continue
            other_pos.append(other.state.p_pos - self.state.p_pos)
            other_vel.append(other.state.p_vel)
        return np.concatenate([self.state.p_vel] + other_pos + other_vel)

    def reward(self, world):
        """Reward for Hostile Bystander for being a threat to the VIP"""
        vip_agent = self.scenario.vip_agent
        rew =  Threat(vip_agent, self.scenario.bodyguards, [self]).calculate_residual_threat_at_every_step()
        bodyguards = self.scenario.bodyguards
        for bodyguard in bodyguards:
            rew += 0.1 * self.distance(bodyguard)
            if self.is_collision(bodyguard):
                rew -= 10
        if self.is_collision(vip_agent):
            rew += 10

        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(self.state.p_pos[p])
            rew -= bound(x)

        return rew
