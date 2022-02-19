import numpy as np

'''
This code returns a list optimal route taken for an agent to get to a targeted location when 
we input the start location and the end location. 

See "optimal_route_image2" file in Image repository for the image 
description of the environment.
'''

gamma = 0.75 # discount factor

alpha = 0.9  #  learning rate

# Define the states (i.e to define the states in numeric form)
location_to_state = {"L1":0, "L2":1, "L3":2, "L4":3, "L5":4, "L6":5}

# Define the rewards
rewards = np.array([[0,0,0,0,1,0],
                    [0,0,0,1,0,1],
                    [0,0,0,1,0,1],
                    [0,1,1,0,1,0],
                    [1,0,0,1,0,0],
                    [0,1,1,0,0,0]])

# Map indices to locations
state_to_location = dict((state,location) for location, state in location_to_state.items())

#This function returns the optimal route yaken to get the targeted location
def optimal_route(start_location, end_location):
    R = rewards.copy()
    Q = np.zeros([R.shape[0], R.shape[0]])

    #Error display when we input start location that is not in the environment
    start_location_error = f"Start location {start_location} is not valid"

    #Error display when we input end location that is not in the environment
    end_location_error = f"End location {end_location} is not valid"

    if start_location not in location_to_state:
        return start_location_error

    if end_location not in location_to_state:
        return end_location_error

    end_state = location_to_state[end_location]

    R[end_state, end_state] = 100

    episodes = 10000

     #Training Phase
    for i in range(episodes):
        current_state = np.random.randint(0, R.shape[0])
        possible_acts = np.where(R[current_state,] >= 1)[0]
        action = int(np.random.choice(possible_acts, size=1))
        max_Q_indices = np.where(Q[action,] == np.max(Q[action,]))[0]
        max_Q_index = int(np.random.choice(max_Q_indices, size=1))
        max_Q = int(Q[action, max_Q_index])
        TD = R[current_state, action] + gamma * max_Q - Q[current_state, action]
        Q[current_state, action] = Q[current_state, action] + alpha * TD

    #Normalize the Q matrix
    Q_normalize = (Q/np.max(Q))*100
    Q = Q_normalize

    #Testing Phase
    initial_state = location_to_state[start_location]
    best_route = [start_location]
    while initial_state != end_state:
        acts = np.where(Q[initial_state,] == np.max(Q[initial_state,]))[0]
        next_state = int(np.random.choice(acts, size=1))
        initial_state = next_state
        best_route.append(state_to_location[next_state])

    print(f"Optimal route from location {start_location} to location {end_location}:")
    return best_route
print("")

print(optimal_route("L2", "L1"))
    
