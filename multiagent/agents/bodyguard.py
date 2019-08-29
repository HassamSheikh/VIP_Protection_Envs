import numpy as np
from . import *

class Bodyguard(Participant):
    """A bodyguard robot"""
    def __init__(self, scenario, communication, alpha=0.5, beta=0.5):
        super().__init__(scenario)
        self.color = np.array([0.0, 0.0, 1.0]) # blue
        self.silent = not communication
        self.state.p_vel = np.zeros(self.scenario.world.dim_p)
        self.state.c = np.zeros(self.scenario.world.dim_c)
        self.allowed_distance = 0.2
        self.alpha=alpha
        self.beta=beta
        self.size = 0.08

    def reset(self):
        super(Bodyguard, self).reset()
        self.state.c = np.zeros(self.scenario.world.dim_c)

    def observation(self):
        """returns the observation of the bodyguard"""
        vip_agent = self.scenario.vip_agent
        bystander_p_pos = np.asarray([bystander.state.p_pos for bystander in self.scenario.bystanders])
        nearest_crowd, crowd_idx = find_nearest_suspected_crowd_members(bystander_p_pos , vip_agent.state.p_pos, 5)
        other_vel = []
        for idx in crowd_idx:
            other_vel.append(self.scenario.bystanders[idx].distance(vip_agent))
        comm = []
        other_agents = []
        if self.scenario.communication:
            for other in self.scenario.bodyguards:
                if other is self: continue
                comm.append(other.state.c)
                other_agents.append(vip_agent.state.p_pos-other.state.p_pos)
        return np.concatenate([self.state.p_vel] + [(vip_agent.state.p_pos-self.state.p_pos)] + [(nearest_crowd - self.state.p_pos).flatten()]+ [other_vel] + other_agents + comm )

    def reward(self, world):
        """Reward for the bodyguards for protecting VIP """
        vip_agent = self.scenario.vip_agent
        residual_threat = Threat(vip_agent, self.scenario.bodyguards, self.scenario.bystanders)
        distance_penalty = 0.5
        if self.allowed_distance <= self.distance(vip_agent) <= residual_threat.safe_distance:
            distance_penalty = 0
        return -(self.alpha*residual_threat.calculate_residual_threat_at_every_step())-(self.beta*distance_penalty)

class TVRBodyguard(Bodyguard):
    def __init__(self, scenario, communication):
        super().__init__(scenario, communication)
        self.color = np.array([0.0, 0.0, 1.0]) # blue
        self.silent = True
        self.state.p_vel = np.zeros(self.scenario.world.dim_p)
        self.state.c = np.zeros(self.scenario.world.dim_c)
        self.allowed_distance = 0.2
        self.action_callback = self.theaction
        self.safe_distance = 0.4

    def theaction(self, agent, world):
        vip = self.scenario.vip_agent
        bodyguards = self.scenario.bodyguards
        bystanders = self.scenario.bystanders
        desired_location = self.quadrant_load_balancing(bystanders, vip, bodyguards, agent)

        # agent_angle = (np.random.uniform(-180.0, 180.0))* np.pi / 180.
        bodyguard_action = Action()
        bodyguard_action.u = (desired_location - agent.state.p_pos)
        #agent.state.p_pos = desired_location#world.agents[0].state.p_pos + np.array([np.cos(agent_angle), np.sin(agent_angle)])*self.allowed_distance
        return bodyguard_action


    def quadrant_load_balancing(self, crowd, vip, bodyguards, agent):
        crowd_in_quadrants = self.place_in_quadrants(crowd, vip)
        updated_location = agent.state.p_pos
        threat_in_each_quadrant = self.calculate_threat_for_all_quadrant(crowd_in_quadrants, vip)
        # for quadrant in threat_in_each_quadrant:
        #     print("Civilians in quadrant "+str(quadrant)+" are "+ str(len(crowd_in_quadrants[quadrant]))+" with threat "+str(threat_in_each_quadrant[quadrant]))
        for _ in range(1, 50):
            max_threat_quadrant = self.quadrant_with_max_threat(threat_in_each_quadrant)
            crowd_in_max_threat_quadrant = crowd_in_quadrants[max_threat_quadrant]
            number_of_crowd_in_max_threat_quadrant = len(crowd_in_max_threat_quadrant)
            for bodyguard in bodyguards:
                if self.is_bodyguard_in_quadrant(bodyguard, vip, max_threat_quadrant):
                    if bodyguard.workload < number_of_crowd_in_max_threat_quadrant:
                        number_of_crowd_in_max_threat_quadrant -= bodyguard.workload
                    else:
                        number_of_crowd_in_max_threat_quadrant = 0
                    break
            if number_of_crowd_in_max_threat_quadrant > 0:
                updated_location = self.calculate_crowd_vectors(vip.state.p_pos, crowd_in_max_threat_quadrant)
                break
            else:
                threat_in_each_quadrant[max_threat_quadrant] = 0
        return updated_location if np.array_equal(updated_location, agent.state.p_pos) else (vip.state.p_pos-agent.state.p_pos)

    def place_in_quadrants(self, crowd, vip):
        quadrant = defaultdict(list)
        for agent in crowd:
            designated_quadrant = self.find_quadrant_based_on_reference(agent, vip)
            quadrant[designated_quadrant].append(self.get_coordinates(agent))
        return quadrant

    def find_quadrant_based_on_reference(self, point_1, reference):
        angle_between_point_1_and_reference = self.calculate_angle_tvr(point_1, reference)
        if 0 <= angle_between_point_1_and_reference < 90:
            return 1
        elif 90 <= angle_between_point_1_and_reference < 180:
            return 2
        elif 180 <= angle_between_point_1_and_reference < 270:
            return 3
        elif 270 <= angle_between_point_1_and_reference < 360:
            return 4

    def calculate_threat_for_all_quadrant(self, crowd_in_quadrants, vip):
        threat_in_quadrants = defaultdict(lambda: 0)
        for quadrant in crowd_in_quadrants:
            if crowd_in_quadrants[quadrant]:
                threat_in_quadrants[quadrant] = self.calculate_threat_in_each_quadrant(crowd_in_quadrants[quadrant], vip.state.p_pos)
            else:
                threat_in_quadrants[quadrant] = 0
        return threat_in_quadrants

    def get_coordinates(self, agents):
        if isinstance(agents, list):
            return np.asarray([agent.state.p_pos for agent in agents])
        return np.asarray(agents.state.p_pos)

    def calculate_distance(self, a, b):
        return np.linalg.norm(a-b)

    def calculate_threat_in_each_quadrant(self, crowd, vip_coordinates):
         threat_level = list(map(lambda x: self.chance_of_attack_tvr(vip_coordinates, x), crowd))
         temp_threat_level = list(map(lambda x: 1.0 - x, threat_level))
         return (1 - reduce(lambda x, y: x*y, temp_threat_level))

    def chance_of_attack_tvr(self, person_1, person_2):
        distance = self.calculate_distance(person_1, person_2)
        if distance > self.safe_distance:
            return 0
        return self.attack_probability(distance)

    def quadrant_with_max_threat(self, all_quadrants):
        return max(all_quadrants, key=all_quadrants.get)

    def is_bodyguard_in_quadrant(self, bodyguard, vip, quadrant):
        return quadrant == self.find_quadrant_based_on_reference(bodyguard, vip)

    def calculate_crowd_vectors(self, vip_coordinates, civilians_in_max_threat_quadrant):
        x, y = 0, 0
        for civilian in civilians_in_max_threat_quadrant:
            angle_between_vip_and_civilian = self.calculate_angle_tvr(civilian, vip_coordinates)
            chance_attack = self.chance_of_attack_tvr(civilian, vip_coordinates)
            magnified_attack = chance_attack * 100
            x = x + math.cos(angle_between_vip_and_civilian) * magnified_attack
            y = y + math.sin(angle_between_vip_and_civilian) * magnified_attack
        desired_location = vip_coordinates + [x, y]
        temp_location = self.find_nearest_location_to_vip(vip_coordinates, desired_location)
        return temp_location


    def find_nearest_location_to_vip(self, vip_coordinates, desired_location):
        desired_angle_from_vip = self.calculate_angle_tvr(desired_location, vip_coordinates)
        return vip_coordinates + [self.allowed_distance * math.cos(desired_angle_from_vip), -self.allowed_distance * math.sin(desired_angle_from_vip)]

    def calculate_angle_tvr(self, agent, reference_point_agent):
        try:
            temp = agent.state.p_pos - reference_point_agent.state.p_pos
        except:
            temp = agent - reference_point_agent
        angle = math.atan2(temp[1], temp[0])
        angle = angle*360/(2*math.pi)
        return angle%360

    def attack_probability(self, distance):
        return np.exp((np.log(0.01)/self.safe_distance)*distance)

    def in_line_of_sight(self, vip, bodyguard, crowd_member):
        return not np.isclose(self.calculate_distance(crowd_member.state.p_pos, vip.state.p_pos), (self.calculate_distance(bodyguard.state.p_pos, vip.state.p_pos) + self.calculate_distance(crowd_member.state.p_pos, bodyguard.state.p_pos)), rtol=0.01)



    def reset(self):
        super(Bodyguard, self).reset()
        self.state.c = np.zeros(self.scenario.world.dim_c)
