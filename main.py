#!/usr/bin/env python3

import sys
import gym
import retro
from model import model
from worker import worker, replay_memory

def main():
    
    
    #env_name = 'CartPole-v0'
    #env_name = 'MountainCar-v0'
    #env_name = 'Acrobot-v1'
    #env_name = 'Pong-v4'
    #env_name = 'PongDeterministic-v4'
    #env = gym.make(env_name)

    #env = retro.make(game='/Users/mainuser/devbin/retro_contest/sonic/',
    #        state='GreenHillZone.Act1', record='./logs')

    #https://github.com/openai/retro/blob/master/retro/retro_env.py#L114
    #env = retro.make(game='Airstriker-Genesis', 
            #use_restricted_actions=retro.ACTIONS_DISCRETE)



    #env = gym.make('Acrobot-v1')
    #env = gym.make('CartPole-v0')
    #env = gym.make('Pong-v4')
    env = gym.make('SpaceInvaders-v0')
    #env = gym.make('Breakout-v0')
    env.seed(0)
    print('[+] environment %s initialized' % '')
    
    #since we are preprocessing state
    state = env.reset()
    n_actions = env.action_space.n #discrete env
    state_shape = worker(None).process_state(state).shape
    print('[*] state shape: %s --> %s\n[*] actions: %s' % (
            state.shape, state_shape, n_actions))
    agent = worker(model(state_shape, n_actions))
    print('[+] worker initialized')

    agent.train(env, episodes=10, print_interval=1)
    agent.test(env, episodes=1, print_interval=1, records=4)

if __name__ == '__main__':
    main()

