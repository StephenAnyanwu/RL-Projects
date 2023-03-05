import random
import numpy as np

# States in the environment. State D and state M are the terminal state.
# The agent wins if in state D and loses if in state M
states = ["A", "B", "C", "D", "E", "F", "G" ,"H", "I", "J", "K", "L", "M", "N"]

#Reward of the states. The reward of terminal states D and M are 5 and -2 respectively.
#The indexs of the rewards indexes correspond to states in the states list/
rewards = [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0]

#Possible actions in each state
actions = ["UP", "DOWN", "LEFT", "RIGHT"]

# Neighboring states of the corresponding states in the states list (0 implies barrier).
state_neighbors = {"A":[0, "E", 0, "B"], "B":[0, "F", "A", "C"], "C":[0, 0, "B", "D"], "D":[0, "G", "C", 0],
                    "E":["A", 0, 0, "F"], "F":["B", "H", "E", 0], "G":["D", "J", 0, 0], "H":["F", "L", 0, "I"],
                    "I":[0, "M", "H", "J"], "J":["G", "N", "I", 0], "K":[0, 0, 0, "L"], "L":["H", 0, "K", "M"],
                    "M":["I", 0, "L", "N"], "N":["J", 0, "M", 0]}

 #Initialized policy
init_policy = {"A":"RIGHT", "B":"RIGHT", "C":"LEFT", "D":"WIN", "E":"UP", "F":"LEFT", "G":"DOWN", 
                "H":"DOWN", "I":"RIGHT", "J":"LEFT", "K":"RIGHT", "L":"RIGHT", "M":"WIN", "N":"UP"}

gamma = 0.98 #discount factor

number_of_episodes = 100 #number of episodes before policy improvement

max_steps = 12 #maximum steps per episode if terminal state not reached

Q_table = np.zeros([len(states), len(actions)]) #Initialize the Q_table to zeros

#Initialized state values
def init_s_values(states):
  s_values = {}
  for state in states:
    if state == "D":
      s_values[state] = 5
      continue
    if state == "M":
      s_values[state] = -2
      continue
    s_values[state] = 0
  return s_values
init_state_values = init_s_values(states)
#print(init_state_values)
      

# Firstly randomly selected state
def first_state(states):
  selected_state = random.choice(states) # randomly select a state
  while selected_state == "D" or selected_state == "M": # repeat selection if the agent selects any of the terminal states
        selected_state
  return selected_state
start_state = first_state(states)
#print(start_state)


#First (s,a) pair selection in random with its associated reward (r) which in turn 
#form the first timestep in an episode.
def first_s_a_r(states, neighbors, actions, rewards, selected_state):
    episode= [] # list that will contain timesteps in an episode
    first_timestep = [] # first timestep in an episode
    selected_state_neighbors = neighbors[selected_state]
    selected_action = random.choice(actions)
    next_state_tran = selected_state_neighbors[actions.index(selected_action)]
    while next_state_tran == 0: # if the agent selects an action that leads to a barrier
        selected_action
    reward = rewards[states.index(next_state_tran)] # reward for action taken
    first_timestep.append(selected_state)
    first_timestep.append(selected_action)
    first_timestep.append(reward)
    episode.append(first_timestep)
    return episode

first_timestep = first_s_a_r(states, state_neighbors, actions, rewards, start_state)
#print(first_timestep)


#This function creates a list of timesteps in an episode while following the policy.
def policy(states, actions, rewards, init_policy, first_timestep, max_steps):
    timesteps = max_steps
    n = 1
    episode = first_timestep
    while  n < timesteps:
        currrent_timestep = episode[-1]
        currrent_timestep_state = currrent_timestep[0]
        currrent_timestep_action = currrent_timestep[1]
        episode_update = episode
        currrent_timestep = []
        action_index = actions.index(currrent_timestep_action)
        current_state_neighbors = state_neighbors[currrent_timestep_state]
        next_state = current_state_neighbors[action_index]
        current_state = next_state
        if current_state == "D" or current_state == "M": # if the agent reaches any of the terminal states
            outcome = init_policy[current_state]
            currrent_timestep.append(current_state)
            currrent_timestep.append(outcome)
            episode_update.append(currrent_timestep)
            episode = episode_update
            return episode
        current_action = init_policy[current_state]
        current_action_index = actions.index(current_action)
        current_state_neighbors = state_neighbors[current_state]
        next_state_tran = current_state_neighbors[current_action_index]
        reward = rewards[states.index(next_state_tran)]
        currrent_timestep.append(current_state)
        currrent_timestep.append(current_action)
        currrent_timestep.append(reward)
        episode_update.append(currrent_timestep)
        episode = episode_update
        n += 1
    return episode

follow_policy = policy(states, actions, rewards, init_policy, first_timestep, max_steps)
#print(follow_policy)


#This function creates a list of unique timesteps in an episode
def firstVisit(complete_episode):
    unique_timesteps = []
    for timestep in complete_episode:
        if timestep not in unique_timesteps:
            unique_timesteps.append(timestep)
    return unique_timesteps

first_visit = firstVisit(follow_policy)
#print(first_visit)


#Calculate the return of of every (s,a) pair in each timestep of an episode
def returns(gamma, first_visit):
    n = 0
    timesteps_G = [timestep for timestep in first_visit if len(timestep) == 3] # remove the terminal state from an episode
    timesteps_rewards = [reward[-1] for reward in timesteps_G] # list of timesteps rewards
    while len(timesteps_rewards) != 0:
        timestep_return = 0
        for t in range(len(timesteps_rewards)):
            timestep_return += gamma**t * timesteps_rewards[t] 
        timesteps_G[n][-1] = timestep_return
        timesteps_rewards.pop(0)
        n += 1
    return timesteps_G

s_a_G = returns(gamma, first_visit)
#print(s_a_G)


#Empty lists of state, action, return (S,A,G) 
def states_actions_returns(states, actions):
    empty_s_a_G = []
    for state in states: 
        empty = []
        for action in actions:
            empty.append([])
        empty_s_a_G.append(empty)
    del(empty_s_a_G[states.index("D")][0:3])
    del(empty_s_a_G[states.index("M")][0:3])
    empty_s_a_G[states.index("D")][0].append("WIN")
    empty_s_a_G[states.index("M")][0].append("LOSE")
    return empty_s_a_G
empty_s_a_G = states_actions_returns(states, actions) #empty list of s,a and G
#print(empty_s_a_G)

#Run multiple episodes and returns a list of Returns(s,a)
def mul_episodes():
    G = states_actions_returns(states, actions)
    for i in range(number_of_episodes):
        start_state = first_state(states)
        first_timestep = first_s_a_r(states, state_neighbors, actions, rewards, start_state)
        follow_policy = policy(states, actions, rewards, init_policy, first_timestep, max_steps)
        first_visit = firstVisit(follow_policy)
        s_a_G = returns(gamma, first_visit)
        for t in s_a_G:
            G[states.index(t[0])][actions.index(t[1])].append(t[2])
    return G
returns_s_a = mul_episodes()
#print(returns_s_a )

#Calculates the average(Returns(s,a))
def average_G(returns_s_a):
    s_a_avaraged_return = empty_s_a_G
    for state in returns_s_a:
        for action in state:
            if "WIN" in action or  "LOSE" in action or len(action) == 0:
                continue
            average_returns = sum(action)/len(action)
            s_a_avaraged_return[returns_s_a.index(state)][state.index(action)].append(average_returns)
    return s_a_avaraged_return
s_a_avg_returns = average_G(returns_s_a)
#print(s_a_avg_returns)


def update_Q_table(s_a_avg_returns):
    for state, i in zip(s_a_avg_returns, range(len(s_a_avg_returns))):
        for action, j in zip(state, range(len(state))):
            if "WIN" in action or  "LOSE" in action or len(action) == 0:
                continue
            Q_table[i,j] = s_a_avg_returns[i][j][0]


def improve_policy():
    updated_policy = init_policy
    for idx, state in enumerate(Q_table):
        if idx == states.index("G") or idx == states.index("M"): #If it's any of the terminal states
            continue
        new_action = actions[np.argmax(state)] #Taking a greedy action
        updated_policy[states[idx]] = new_action
    init_policy = updated_policy


        
'''---------------------------Training Phase-----------------------------------'''
    
for i in range(10000):
    returns_s_a = mul_episodes()
    s_a_avg_returns = average_G(returns_s_a)
    update_Q_table(s_a_avg_returns)
    improve_policy()
    
for state in init_state_values:
  init_state_values[state] = np.max(Q_table)

optimized_polcy = init_policy
print(Q_table) #The converged Q(s,a)
print(optimized_polcy)



'''-----------------------------Testing Phase---------------------------------------'''


starting_state = random.choice(states)
while starting_state  == "D" or starting_state  == "M": 
    starting_state
optimal_path = [starting_state]
current_state = starting_state
while optimal_path[-1] != "D":
    best_action = actions[np.argmax(Q_table[states.index(current_state)])] #The optimal action in current state
    best_action_idx = actions.index(best_action)
    next_state = state_neighbors[current_state][best_action_idx]
    optimal_path.append(next_state)
    current_state = next_state

print(f"Optimal path:\n {optimal_path}")
