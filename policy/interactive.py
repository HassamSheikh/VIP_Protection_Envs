#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import pandas as pd
import gc
from collections import defaultdict
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy, PSController
import multiagent.scenarios as scenarios

def state_information(world, episode_number, action_n, episode_data):
    axis= ['x', 'y']
    episode_data["episode"].append(episode_number)
    for agent in world.agents:
        for idx, p_pos in enumerate(agent.state.p_pos):
            key_name = agent.name + "_p_pos_" + axis[idx]
            episode_data[key_name].append(p_pos)

        for idx, p_vel in enumerate(agent.state.p_vel):
            key_name = agent.name + "_p_vel_" + axis[idx]
            episode_data[key_name].append(p_vel)

    for idx, agent in enumerate(world.policy_agents):
        episode_data[agent.name + "_actions"].append([agent.action.u])
        episode_data[agent.name + "_raw_actions"].append([act_n[idx]])


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None, allow_abbrev=False)
    parser.add_argument("--data-path", default='./data/trajectory_data.csv', help="Path for saving trajectories")
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--save-rate", type=int, default=10, help="Save rate to store trajectories to file")
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    try:
        data = pd.read_csv(args.data_path)
        episode = data.iloc[-1]["episode"] + 1
        del [data]
        gc.collect()
    except:
        episode = 1
    data = pd.DataFrame()
    print("Current Episode Number {}".format(episode))
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done, info_callback=None, shared_viewer = True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [PSController(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    episode_data = defaultdict(list)
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        state_information(env.world, episode, act_n, episode_data)
        if any(done_n):
            print("Current Episode Number {}".format(episode))
            obs_n = env.reset()
            temp = pd.DataFrame(episode_data)
            data = data.append(temp, ignore_index=False)
            episode_data = defaultdict(list)
            if episode % args.save_rate == 0:
                header = (episode == 1)
                with open(args.data_path, 'a') as f:
                    data.to_csv(f, header=header)
                del [data]
                gc.collect()
                data = pd.DataFrame()
            episode += 1

        # render all agent views
        env.render()

        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
