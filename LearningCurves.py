# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:00:55 2023

@author: jmd01
"""


from MyLearners import Learner,RWLearner
from Raw_input import Raw_input, ProbabilisticGrammar
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures as cf
from datetime import datetime
#import pandas as pd
start_time = datetime.now()
import scipy

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Helvetica"
})

def logistic(x,k,x0):
    return  1/(1+np.exp(-k*(x-x0))) 


def run_simulation(l):
    l.learn(stimuli_stream)
    return l

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

    
def get_success_and_length(learners):
    success = 0.*np.array(learners[0].success)
    sent_len = 0.*np.array(learners[0].sent_len)
    for l in learners:
        success += np.array(l.success)
        sent_len += np.array(l.sent_len)
        
    success /= n_sim
    sent_len /= n_sim
    return success, sent_len

def get_averaged_final_index(learners):
    final = 0
    for l in learners:
        final += l.final_index
        
    return final/len(learners)
    
def plot_learning_curve(learnersC,learnersN,RWlearnersC,RWlearnersN):
    # Process the result   
    colors = ['b','g','m','c']
    ll = [learnersC,learnersN,RWlearnersC,RWlearnersN]
    lab = ['Q-learner, cont','Q-learner, next','RW Q-learner, cont', 'RW Q-learner, next']
    for i in range(4):
        success, sent_len = get_success_and_length(ll[i])
    
        trial_vec = range(ll[i][0].n_trials+1)
        x_data = np.linspace(0,len(success),len(success))
        y_data = success
        popt,pcov = scipy.optimize.curve_fit(logistic,x_data,y_data,maxfev=10000)
        print('Optimal parameters')
        print(popt)
        print('learning time')
        print(2*popt[-1])
        print(np.diag(pcov)[-1])
        plt.plot(x_data,logistic(x_data,*popt),'k:', label='_nolegend_')
        #plt.axvline(x = 2*popt[-1],color = 'k')
        # Plot the results
        plt.scatter(trial_vec,success,s = 2,c=colors[i],label = lab[i])
        plt.xlabel('Number of trials')
        plt.ylabel('Fraction of correct responses')
    #plt.colorbar()
    plt.legend()
    plt.savefig('LearningCurves.pdf')
    plt.show()
    
    


###############################################################
#
#       Setting learning parameters
#
###############################################################

# Set the initial learning parameters
Learner.initial_value_border = 1.
Learner.initial_value_chunking = -1.
Learner.ID = 0

# Set the parameters controlling reinforcement learning
Learner.alpha = 0.1
Learner.beta = 1.9
Learner.positive_reinforcement = 25.
Learner.negative_reinforcement = -10.

RWLearner.alpha = 0.1
RWLearner.beta = 1.9
RWLearner.positive_reinforcement = 25.
RWLearner.negative_reinforcement = -10.


###############################################################
#
#       Definition of the language using a probabilistic CFG
#
###############################################################

print('Defining the grammar')

# definition of the grammar
# Vocabulary
number_of_verbs = 5
number_of_nouns = 5
number_of_adj = 0
number_of_relpron = 2
number_of_det = 0

verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
det = ['d' + str(i) for i in range(1, number_of_det+1)]

terminals2 = flatten([verbs,nouns,adjs,relpron,det])
non_terminals2 = ['S', 'N','NP','VP','V','RelCl']

###############################################################
#
#       NVN language
#
###############################################################

# Grammatical rules
production_rulesNVN = {
    'S': [['N', 'VP','N']],
    'VP': [['V']],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
}

weightsNVN = {
    'S': [1],
    'VP': [1],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)]    
    }

cfgNVN = ProbabilisticGrammar(terminals2, non_terminals2, production_rulesNVN,weightsNVN)
# Context free grammar


###############################################################
#
#       NVN + relative clauses with relative pronouns language
#
###############################################################

# Grammatical rules
production_rulesRCP = {
    'S': [['N', 'VP'],['N','VP','RelCl'],['N','VP','RelCl','RelCl']],
    'VP': [['V','N'],['V']],
    'RelCl': [['r' + str(i) , 'VP'] for i in range(1, number_of_relpron+1)],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
}

weightsRCP = {
    'S': [0.5, 0.25 ,0.25 ],
    'VP': [.5, .5],
    'RelCl': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)]    
    }

relweight = np.array([1/2**i for i in range(len(relpron))])
relweight /= np.sum(relweight)

nweight = np.array([1/2**i for i in range(len(nouns))])
nweight /= np.sum(nweight)

vweight = np.array([1/2**i for i in range(len(verbs))])
vweight /= np.sum(vweight)

weightsRCPZipf = {
    'S': [0.5, 0.25 ,0.25 ],
    'VP': [.5, .5],
    'RelCl': relweight,
    'N': nweight,
    'V': vweight   
    }

# Context free grammar
cfgRCP = ProbabilisticGrammar(terminals2, non_terminals2, production_rulesRCP,weightsRCPZipf)


###############################################################
#
#       NVN + relative clauses without relative pronouns language
#
###############################################################

# Grammatical rules
production_rulesRC = {
    'S': [['N', 'VP'],['N','VP','RelCl'],['N','VP','RelCl','RelCl']],
    'VP': [['V','N']],
    'RelCl': [['VP'] ],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
}

weightsRC = {
    'S': [0.5, 0.25 ,0.25 ],
    'VP': [1],
    'RelCl': [1],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)]    
    }

# Context free grammar
cfgRC = ProbabilisticGrammar(terminals2, non_terminals2, production_rulesRC,weightsRC)

#############################################################
#
#       Yang and Piantadosi grammar
#
#############################################################

number_of_verbs = 1
number_of_nouns = 1
number_of_adj = 1
number_of_relpron = 1
number_of_det = 1
number_of_prep = 1

verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
det = ['d' + str(i) for i in range(1, number_of_det+1)]
prep = ['p' + str(i) for i in range(1, number_of_prep+1)]


terminalsYP = flatten([verbs,nouns,adjs,relpron,det,prep])
non_terminalsYP = ['S','NP','VP','AP','PP','N','V','A','D','P','rel']

production_rulesYP = {
    'S': [['NP', 'VP']],
    'NP': [['N'],['D','N'],['D','AP','N'],['NP','PP']],
    'VP': [['V'],['V','NP'],['V','rel','S'],['VP','PP']],
    'AP': [['A'],['A','AP' ] ],
    'PP': [['P','NP']],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)],
    'A': [['a' + str(i)] for i in range(1, number_of_adj+1)],
    'D': [['d' + str(i)] for i in range(1, number_of_det+1)],
    'P': [['p' + str(i)] for i in range(1, number_of_prep+1)],
    'rel': [['r' + str(i)] for i in range(1, number_of_relpron+1)]
}

weightsYP = {
    'S': [1.0],
    'NP': [.25,.25,.25,.25],
    'VP': [.25,.25,.25,.25],
    'AP': [.5,.5 ],
    'PP': [1],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)],
    'A': [1/number_of_adj for i in range(1, number_of_adj+1)],
    'D': [1/number_of_det for i in range(1, number_of_det+1)],
    'P': [1/number_of_prep for i in range(1, number_of_prep+1)],
    'rel': [1/number_of_relpron for i in range(1, number_of_relpron+1)]  
    }

cfgYP = ProbabilisticGrammar(terminalsYP, non_terminalsYP, production_rulesYP,weightsYP)

#############################################################
#
#       Initializing the learners
#
#############################################################

# number of simulations
n_sim = 100
n_trials = 2000

#############################################################
#
#   Creating the stimuli stream
#
#############################################################
print('Creating the stimuli stream')

# Create stimuli stream
stimuli_stream = Raw_input(3*n_trials,cfgNVN)

print('Initializing learners')

# Select type of chunking mechanism
#typ = 'right'
typ = 'flexible'
border = 'nxt'

# Create as many learners as number of simulations.
learnersC = [Learner(n_trials = n_trials, border = 'cont') for i in range(n_sim)]
learnersN = [Learner(n_trials = n_trials, border = 'next') for i in range(n_sim)]
RWlearnersC = [RWLearner(n_trials = n_trials, border = 'cont') for i in range(n_sim)]
RWlearnersN = [RWLearner(n_trials = n_trials, border = 'next') for i in range(n_sim)]

#learner = RWLearner(n_trials = n_trials, border = border)
#learner.learn_with_snapshot(stimuli_stream, 'test.xlsx', [1000,2000,3000,4000], 5)


#############################################################
#
#       Running the simulation in parallel
#
#############################################################    
print('Running the simulation in parallel')

print('Q-learning with continuous border')
# # Run the simulation in parallel
with cf.ThreadPoolExecutor() as executor:
    print("Number of worker threads:", executor._max_workers)
    results = [executor.submit(run_simulation, l) for l in learnersC]
    
    # Iterate over the results as they become available
    for future in cf.as_completed(results):
        result = future.result()
        # Combine the result with other results as necessary
        
# # Run the simulation in parallel
print('Q-learning with next sentence condition')
with cf.ThreadPoolExecutor() as executor:
    print("Number of worker threads:", executor._max_workers)
    results = [executor.submit(run_simulation, l) for l in learnersN]
    
    # Iterate over the results as they become available
    for future in cf.as_completed(results):
        result = future.result()
        
# # Run the simulation in parallel
print('RW Q-learning with continuous condition')
with cf.ThreadPoolExecutor() as executor:
    print("Number of worker threads:", executor._max_workers)
    results = [executor.submit(run_simulation, l) for l in RWlearnersC]
    
    # Iterate over the results as they become available
    for future in cf.as_completed(results):
        result = future.result()
        
# # Run the simulation in parallel
print('RW Q-learning with next sentence condition')
with cf.ThreadPoolExecutor() as executor:
    print("Number of worker threads:", executor._max_workers)
    results = [executor.submit(run_simulation, l) for l in RWlearnersN]
    
    # Iterate over the results as they become available
    for future in cf.as_completed(results):
        result = future.result()
    
#############################################################
#
#       Postprocessing
#
#############################################################
print('Postprocessing')


plot_learning_curve(learnersC,learnersN,RWlearnersC,RWlearnersN)
print(get_averaged_final_index(learnersC))
print(get_averaged_final_index(learnersN))
print(get_averaged_final_index(RWlearnersC))
print(get_averaged_final_index(RWlearnersN))


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

