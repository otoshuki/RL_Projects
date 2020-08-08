#Visualize - Step 2
def get_action_value(mdp, state_values, state, action, gamma):
    """ Compute Q(s,a) """
    #Summation over all the next states
    q_func = 0
    for state_p in mdp.get_next_states(state, action):
        model_prob = mdp.get_transition_prob(state, action, state_p)
        reward = mdp.get_reward(state,action,state_p)
        q_func += model_prob*(reward+gamma*state_values[state_p])    
    return q_func    
