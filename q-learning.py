import numpy as np
import random



"""
A dog is left in an environment, he wants to rest in a particular location (L6) within that environment.
The only option given for him to rest in that particular location is to go to that location and pick up 
a dead rat and move it to another designated location then (L1) come back to that resting location and have 
his rest.
"""

R = np.array([[[-1,-1,-1,-1,0,-1],
              [-1,-1,-1,0,-1,100],
              [-1,-1,-1,0,-1,100],
              [-1,0,0,-1,0,-1],
              [0,-1,-1,0,-1,-1],
              [-1,0,0,-1,-1,-1]],
            [[-1,-1,-1,-1,0,-1],
              [-1,-1,-1,0,-1,0],
              [-1,-1,-1,0,-1,0],
              [-1,0,0,-1,0,-1],
              [100,0,-1,0,-1,-1],
              [-1,0,0,-1,-1,-1]]])


Q = np.zeros((2,6,6))

gamma = 0.9

current_state = 1

def available_actions(state):
    possible_index = []
    for i in range(0,2):
        current_state_row = R[i, state]
        possible_actions = np.where(current_state_row>=0)[0]
        possible_index.append(list(possible_actions))
    return possible_index

def sample_next_action(available_action_range):
    samples = available_action_range
    next_actions = []
    for i in samples:
        act = random.choice(i)
        next_actions.append(act)
    return next_actions

def max_Q_index(action):
    max_index = []
    for i in range(2):
        index_of_action = np.where(Q[i, action[i],] == np.max(Q[i, action[i]]))[0]
        selected_index = int(np.random.choice(index_of_action, size=1))
        max_index.append(selected_index)
    return max_index

def max_Q_value(action, max_index):
    max_Q = []
    for i in range(0,2):
        max_Qs =int(Q[i, action[i], max_index[i]])
        max_Q.append(max_Qs)
    return max_Q
    
def update(current_state, action, gamma, max_Q_value):
    for i in range(0,2):
        Q[i,current_state, action[i]] = R[i, current_state, action[i]] + (gamma * max_Q_value[i])

for i in range(10000):
    current_state = np.random.randint(0, 6)
    possible_action = available_actions(current_state)
    action = sample_next_action(possible_action)
    max_Q_indexes = max_Q_index(action)
    max_Q_values = max_Q_value(action, max_Q_indexes)
    update(current_state, action, gamma, max_Q_values)


print("--------------------------------------------------------------------------------------")

'''For a dog(agent) to rest in L1(location 1) firstly, it has to go to L1 from any location and move a died rat to L1(Location 1) and come 
back to L1 to have its rest.'''

state_label = {0:"L1", 1:"L2", 2:"L3", 3:"L4", 4:"L5", 5:"L6"}
location_label = dict((location, state) for state, location in state_label.items())

current_state = int(np.random.randint(0,6))

#From L6 to L1 to drop the rat
if current_state == 5:
    current_location = current_state
    optimal_policy = [state_label[current_location]]
    while current_location != 0:
        max_index = np.where(Q[1, current_location,] == np.max(Q[1, current_location,]))[0]
        next_state = int(np.random.choice(max_index))
        current_location = next_state
        optimal_policy.append(state_label[current_location])
    print(f"From {optimal_policy[0]} to {optimal_policy[-1]} to drop the dead rat.")
    print(f"Route taken: {optimal_policy}")
    print("")

    #From L1 back to L6 to rest
    if current_location == 0:
        current_state = 0
        current_location = current_state
        optimal_policy = [state_label[current_location]]
        while current_location != 5:
            max_index = np.where(Q[0, current_location,] == np.max(Q[0, current_location,]))[0]
            next_state = int(np.random.choice(max_index))
            current_location = next_state
            optimal_policy.append(state_label[next_state])
        print(f"From {optimal_policy[0]} back to {optimal_policy[-1]} to rest.")
        print(f"Route taken: {optimal_policy}")
        print("")
else:
    #From any location  to L6
    current_location = current_state
    optimal_policy = [state_label[current_location]]
    while current_location != 5:
        max_index = np.where(Q[0, current_location,] == np.max(Q[0, current_location,]))[0]
        next_state = int(np.random.choice(max_index))
        current_location = next_state
        optimal_policy.append(state_label[current_location])
    print(f"From {optimal_policy[0]} to {optimal_policy[-1]} to pick up the dead rat.")
    print(f"Route taken: {optimal_policy}")
    print("")

    #From L6 to L1 to drop the rat
    if current_location == 5:
        current_state = 5
        current_location = current_state
        optimal_policy = [state_label[current_location]]
        while current_location != 0:
            max_index = np.where(Q[1, current_location,] == np.max(Q[1, current_location,]))[0]
            next_state = int(np.random.choice(max_index))
            current_location = next_state
            optimal_policy.append(state_label[current_location])
        print(f"From {optimal_policy[0]} to {optimal_policy[-1]} to drop the dead rat.")
        print(f"Route taken: {optimal_policy}")
        print("")

        #From L1 back to L6 to rest
        if current_location == 0:
            current_state = 0
            current_location = current_state
            optimal_policy = [state_label[current_location]]
            while current_location != 5:
                max_index = np.where(Q[0, current_location,] == np.max(Q[0, current_location,]))[0]
                next_state = int(np.random.choice(max_index))
                current_location = next_state
                optimal_policy.append(state_label[next_state]) 
            print(f"From {optimal_policy[0]} back to {optimal_policy[-1]} to rest.")
            print(f"Route taken: {optimal_policy}")
            print("")
            


        


