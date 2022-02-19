import numpy as np
import random

'''
In this project, there are two agents placed in an environment. They will find an optimal route (placed in list) to a particular location, 
pick up an item then take it to a designated drop off location (placed in a list) while following an optimal route still within the 
environment.
If both agents are already in the pick up location, both are randomly selected to take the item to the drop off location while taking the 
optimal route.
If either of the agents is already in the pick up location, that agent will take the item to the drop off location while taking the 
optimal route.
Any agent closer to the pick up location will perform the duty while taking the optimal routes. 
If both agents are placed in the same location (not the pick up location) or if their respective length of optimal route list are equal, 
both are randomly selected to perform the duty.


'''


gamma = 0.75 #discount factor

alpha = 0.9 #learning rate
          
R_matrix = np.array([
             [[0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
              [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
              [1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
              [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1],
              [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0],
              [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0],
              [0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0]],
              [[0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
              [1,0,100,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
              [1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
              [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1],
              [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0],
              [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
              [0,0,100,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0],
              [0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0]]
              ])

Q = np.zeros((R_matrix.shape[0],R_matrix.shape[1],R_matrix.shape[2]))


location_to_state =  {"L1":0, "L2":1, "L3":2, "L4":3, "L5":4, "L6":5, "L7":6, "L8":7, "L9":8, "L10":9, 
                        "L11":10, "L12":11, "L13":12, "L14":13,"L15":14, "L16":15, "L17":16, "L18":17}

state_to_location = dict((state, location) for location, state in location_to_state.items())

class TWoAgents:
    
    def __init__(self, pickup_loc):
        self.pickup_loc = pickup_loc # The location where item is to be picked

    def __repr__(self):
        if self.pickup_loc not in location_to_state:
             return (f"______Error______\nLocation {self.pickup_loc} not in the environment.")
        else:
            return (f"Location {self.pickup_loc} is in the environment ")

    def update_R(self):
        '''
        This method returns updated Ratrix. The matrix is updated based on the pick up location inputted.
        '''
        R = R_matrix.copy()
        duty_state = location_to_state[self.pickup_loc]
        pickup_loc_exit = list(np.where(R[0,duty_state] > 0)[0])
        for i in pickup_loc_exit:
            R[0,i,duty_state] = 100
        return R
   
    def trained_Q(self):
        '''
        This method returns the train Q matrix.
        '''
        if self.pickup_loc not in location_to_state:
            return f"______Error______\nLocation {self.pickup_loc} not in the environment."
        episodes = 10000
        R = self.update_R()
        for i in range(episodes):
            
            current_state = np.random.randint(0, R_matrix.shape[1])
            possible_index = []
            for i in range(R_matrix.shape[0]):
                current_state_row = R[i, current_state,]
                possible_actions = np.where(current_state_row > 0)[0]
                possible_index.append(list(possible_actions))

            samples = possible_index
            next_actions = []
            for i in samples:
                act = random.choice(i)
                next_actions.append(act)

            action = next_actions
            max_index = []
            for i in range(R_matrix.shape[0]):
                index_of_action = np.where(Q[i, action[i],] == np.max(Q[i, action[i]]))[0]
                selected_index = int(np.random.choice(index_of_action, size=1))
                max_index.append(selected_index)

            max_Q = []
            for i in range(R_matrix.shape[0]):
                max_Qs =int(Q[i, action[i], max_index[i]])
                max_Q.append(max_Qs)

            max_Q_value = max_Q
            for i in range(R_matrix.shape[0]):
                TD = R[i, current_state, action[i]] + gamma * max_Q_value[i] - Q[i,current_state, action[i]]
                Q[i, current_state, action[i]] += alpha + TD
        
        normalized_Q = (Q/np.max(Q))*100
        return normalized_Q
            
    def perform_duty(self, agent_1_loc, agent_2_loc):
        '''
        This method returns the optimal policy taken to perform tasks.
        '''
        if self.pickup_loc not in location_to_state:
            return (f"______Error______\n Pickup location {self.pickup_loc} not in the environment.")
        if agent_1_loc not in location_to_state:
            return (f"Error: Agent 1 location {agent_1_loc} not in the environment.")
        if agent_2_loc not in location_to_state:
            return (f"Error: Agent 1 location {agent_1_loc} not in the environment.")
        Q = self.trained_Q()

        # Agent 1 optimal route to pick up location
        initial_state = location_to_state[agent_1_loc]
        agent_1_route_to_pickup = [agent_1_loc]
        while initial_state != location_to_state[self.pickup_loc]:
            acts = np.where(Q[0, initial_state,] == np.max(Q[0, initial_state,]))[0]
            next_state = int(np.random.choice(acts, size=1))
            agent_1_route_to_pickup.append(state_to_location[next_state])
            initial_state = next_state

        # Agent 2 optimal route to optimal location
        initial_state = location_to_state[agent_2_loc]
        agent_2_route_to_pickup =[agent_2_loc]
        while initial_state != location_to_state[self.pickup_loc]:
            acts = np.where(Q[0, initial_state,] == np.max(Q[0, initial_state,]))[0]
            next_state = int(np.random.choice(acts, size=1))
            agent_2_route_to_pickup.append(state_to_location[next_state])
            initial_state = next_state

        # Drop off
        drop_off_location = "L3"
        drop_off_state =location_to_state[drop_off_location]
        equal_lenght = len(agent_1_route_to_pickup) == len(agent_2_route_to_pickup)
        equal_route = agent_1_route_to_pickup == agent_2_route_to_pickup

        if agent_1_loc == self.pickup_loc and agent_2_loc == self.pickup_loc:
            selected_agent = random.choice([agent_1_loc, agent_2_loc])
            initial_state = location_to_state[selected_agent]
            drop_off_route = [selected_agent]
            while initial_state != drop_off_state:
                acts = np.where(Q[1, initial_state,] == np.max(Q[1, initial_state,]))[0]
                next_state = int(np.random.choice(acts, size=1))
                drop_off_route.append(state_to_location[next_state])
                initial_state = next_state
            report = f"Both agents are in the pick up location {selected_agent}"
            if selected_agent == agent_1_loc:
                duty = f"Agent 1 decides to perform \nOptimal route taken to drop off the item in location {drop_off_location}: {drop_off_route}"
                return (f"{report}\n{duty}")
            else:
                duty = f"Agent 2 decides to perform \nOptimal route taken to drop off the item in location {drop_off_location}: {drop_off_route}"
                return(f"{report}\n{duty}")

        if agent_1_loc == self.pickup_loc:
            initial_state = location_to_state[agent_1_loc]
            drop_off_route = [agent_1_loc]
            while initial_state != drop_off_state:
                acts = np.where(Q[1, initial_state,] == np.max(Q[1, initial_state,]))[0]
                next_state = int(np.random.choice(acts, size=1))
                drop_off_route.append(state_to_location[next_state])
                initial_state = next_state
            duty = f"Agent 1 is in the pick up location {self.pickup_loc} \nOptimal route taken to drop off the item in location {drop_off_location}: {drop_off_route}"
            return duty

        if agent_2_loc == self.pickup_loc:
            initial_state = location_to_state[agent_2_loc]
            drop_off_route = [agent_2_loc]
            while initial_state != drop_off_state:
                acts = np.where(Q[1, initial_state,] == np.max(Q[1, initial_state,]))[0]
                next_state = int(np.random.choice(acts, size=1))
                drop_off_route.append(state_to_location[next_state])
                initial_state = next_state
            duty = f"Agent 1 is in the pick up location {self.pickup_loc} \nOptimal route taken to drop off the item in location {drop_off_location}: {drop_off_route}"
            return duty

        if equal_lenght == True and equal_route == False:
            selected_agent = random.choice([agent_1_route_to_pickup,agent_2_route_to_pickup,])
            initial_location = selected_agent[-1]
            initial_state = location_to_state[initial_location]
            drop_off_route = [initial_location]
            while initial_state != drop_off_state:
                acts = np.where(Q[1, initial_state,] == np.max(Q[1, initial_state,]))[0]
                next_state = int(np.random.choice(acts, size=1))
                drop_off_route.append(state_to_location[next_state])
                initial_state = next_state
            report = f"Pick up location {self.pickup_loc} is closer to Agent 1 and Agent 2"
            agent_1_duty = f"Agent 1 who is in location {agent_1_loc} decides to perform.\nOptimal route taken to pick up item in location {self.pickup_loc}: {agent_1_route_to_pickup} \nOptimal route taken to drop off item in location {drop_off_location}: {drop_off_route} "
            agent_2_duty = f"Agent 2 who is in location {agent_2_loc} decides to perform.\nOptimal route taken to pick up item in location {self.pickup_loc}: {agent_2_route_to_pickup} \nOptimal route taken to drop off item in location {drop_off_location}: {drop_off_route} "
            if selected_agent == agent_1_route_to_pickup:
                return (f"{report} \n{agent_1_duty}")
            return (f"{report} \n{agent_2_duty}")

        if equal_lenght==True and equal_route ==True:
            selected_agent = random.choice([agent_1_route_to_pickup, agent_2_route_to_pickup])
            initial_location = selected_agent[-1]
            initial_state = location_to_state[initial_location]
            drop_off_route = [initial_location]
            while initial_state != drop_off_state:
                acts = np.where(Q[1, initial_state,] == np.max(Q[1, initial_state,]))[0]
                next_state = int(np.random.choice(acts, size=1))
                drop_off_route.append(state_to_location[next_state])
                initial_state = next_state

            report = f"Agent 1 and Agent 2 are in the same location {agent_1_loc}"
            agent_1_duty = f"Agent 1 decides to perform.\nOptimal route taken to pick up item in location {self.pickup_loc}: {agent_1_route_to_pickup} \nOptimal route taken to drop off item in location {drop_off_location}: {drop_off_route} "
            agent_2_duty = f"Agent 2 decides to perform.\nOptimal route taken to pick up item in location {self.pickup_loc}: {agent_2_route_to_pickup} \nOptimal route taken to drop off item in location {drop_off_location}: {drop_off_route} "
            if selected_agent == agent_1_route_to_pickup:
                return (f"{report}\n{agent_1_duty}")
            else:
                return (f"{report}\n{agent_2_duty}")

        if len(agent_1_route_to_pickup) < len(agent_2_route_to_pickup):
            initial_location = agent_1_route_to_pickup[-1]
            initial_state = location_to_state[initial_location]
            drop_off_route = [initial_location]
            while initial_state != drop_off_state:
                acts = np.where(Q[1, initial_state,] == np.max(Q[1, initial_state,]))[0]
                next_state = int(np.random.choice(acts, size=1))
                drop_off_route.append(state_to_location[next_state])
                initial_state = next_state
            
            report = f"Agent 1 who is in location {agent_1_loc} is more closer to pick up location {self.pickup_loc}"
            agent_1_duty = f"Agent 1 optimal route taken to pick up item in location {self.pickup_loc}: {agent_1_route_to_pickup} \nOptimal route taken to drop off item in location {drop_off_location}: {drop_off_route}"  
            return (f"{report}\n{agent_1_duty}")

        else:
            initial_location = agent_2_route_to_pickup[-1]
            initial_state = location_to_state[initial_location]
            drop_off_route = [initial_location]
            while initial_state != drop_off_state:
                acts = np.where(Q[1, initial_state,] == np.max(Q[1, initial_state,]))[0]
                next_state = int(np.random.choice(acts, size=1))
                drop_off_route.append(state_to_location[next_state])
                initial_state = next_state
            report = f"Agent 2 who is in location {agent_2_loc} is more closer to pick up location {self.pickup_loc}"
            agent_2_duty = f"Agent 2 optimal route taken to pick up item in location {self.pickup_loc}: {agent_2_route_to_pickup} \nOptimal route taken to drop off item in location {drop_off_location}: {drop_off_route}"  
            return (f"{report}\n{agent_2_duty}")

        
        

pickup_loc =TWoAgents("L7")
perform = pickup_loc.perform_duty("L2", "L5")
print(perform)






         

           
    


        

    
    





