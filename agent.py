import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.epsilon = 0.005
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 1
        self.gamma = 1.0
        
    def get_policy_for_state(self, state):
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        policy_s[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon / self.nA)
        return policy_s
    
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy = self.get_policy_for_state(state)
        action = np.random.choice(np.arange(self.nA), p=policy)

        return action
#         return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        policy_s = self.get_policy_for_state(next_state)
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma*np.dot(policy_s, self.Q[next_state]) - self.Q[state][action])

#         self.Q[state][action] += 1