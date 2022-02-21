import random
import numpy as np

'''
In this RL project, the agent randomly chooses a targeted location while we choose a starting location in an input prompt for the agent and the 
agent returns an optimal route to the targeted location in a list. optimal route is return if the locations (both starting and targeted) are 
situated in the environment. 
See  optimal_route_image1.png in Image repository  file that contains the diagram description of the environment. 
'''

gamma = 0.75 # discount factor

alpha = 0.9  #  learning rate
#D Define the actions (transition to the next state)
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

 # Define the rewards
R = np.array([[0,1,0,0,0,0,0,0,0],
              [1,0,1,0,1,0,0,0,0],
              [0,1,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,1,0,0],
              [0,1,0,0,0,0,0,1,0],
              [0,0,1,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,1,0],
              [0,0,0,0,1,0,1,0,1],
              [0,0,0,0,0,0,0,1,0]])
            
Q = np.zeros([R.shape[0], R.shape[0]])


# Define the states (i.e to define the states in numeric form)
location_to_state = {"L1":0, "L2":1, "L3":2, "L4":3, "L5":4, "L6":5, "L7":6, "L8":7, "L9":8}

 # Map indices to locations
state_to_location = dict((state,location) for location, state in location_to_state.items())

# This function assigns a reward value to targeted location in R matrix 
def update_R(end_location):
    end_state = location_to_state[end_location]
    R[end_state, end_state] = 999
    return R

# Agent randomly selects targeted location
targets = [i for i in location_to_state]
targeted_location = random.choice(targets)

# Map indices to state
targeted_state = location_to_state[targeted_location]


R = update_R(targeted_location)
    
# This function returns all possible actions in the state given as an argument
def available_acts(state):
    current_state = R[state,]
    possible_actions = np.where(current_state >= 1)[0]
    return possible_actions

# This function chooses at random which action to be performed within the range of all the available actions
def next_action(available_acts):
    act = int(np.random.choice(available_acts, size=1))
    return act

# This function updates the Q matrix according to the path selected and the Q-Learning algorithm
def update_Q(current_state, action, gamma, alpha): 
    max_Q_index = np.where(Q[action,] == np.max(Q[action,]))[0]
    max_Q_index = int(np.random.choice(max_Q_index, size=1))
    max_Q = int(Q[action, max_Q_index])
    TD = R[current_state, action] + gamma * max_Q - Q[current_state, action]
    Q[current_state, action] += alpha + TD

# This function returns trained Q matrix over 10000 iteration(episodes)
def train():
    for i in range(10000):
        current_state = int(np.random.randint(0, R.shape[0])) 
        possible_actions = available_acts(current_state)
        action = next_action(possible_actions)
        update_Q(current_state, action, gamma, alpha)
    return Q

# Trained Q matrix
Q = train()

#Normalized trained Q matrix
Q = (Q/np.max(Q))*100

#Testing Phase
print("")
print("-------------------------------------------------------------------------------------------")
name = (input("Welcome, please enter your name: ")).capitalize()
print("")
print(f"Hello {name}, my name is Agent Snow. My targeted location is {targeted_location}. Which location do you want me to started from?")

for i in range(1,5):
    current_location = (input("Input starting location: ")).upper()
    print("")

    if current_location == targeted_location:

        if i == 2:
            print(f"       Invalid Selection!!! \n{current_location} is the targeted location [2 chances left].")
            print("")
            print(f"{name}, my targeted location is {targeted_location}. Which location do you want me to started from?")
            print("")
        else:
            if i == 4:
                print(f"     Invalid Selection!!! \n{name}, {current_location} is the targeted location")
                print("Request Terminated")
                print("-------------------------------------------------------------------------------------------")
            else:
                 print(f"    Invalid Selection!!! \n{name}, {current_location} is the targeted location")
                 print("")

    else:
        if (current_location in location_to_state) and (current_location != targeted_location):
            current_state = location_to_state[current_location]
            end_state = targeted_state
            best_route = [state_to_location[current_state]]
            while current_state != end_state:
                max_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[0]
                next_state = int(np.random.choice(max_index))
                current_state = next_state
                best_route.append(state_to_location[next_state])

            print(f"{name}, best route taken to get to location {targeted_location} is... ")
            print(best_route)
            print("-------------------------------------------------------------------------------------------")
            break

        else:
        
            if i == 2:
                print(f"{name}, location {current_location} not found.\nPlease try again [2 chances left] ")
                print("")
                print(f"{name}, my targeted location is {targeted_location}, which location do you want me to started from?")
                print("")
            else:
                if i == 4:
                    print(f"{name}, location {current_location} not found")
                    print("Request Terminated")
                    print("-------------------------------------------------------------------------------------------")
                else:    
                    print(f"{name}, location {current_location} not found.\nPlease try again")
                    print("")

            
