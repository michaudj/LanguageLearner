# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:27:51 2023

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

def success_by_length(learners):
    filtered_success = {}
    length_filtered = {}
    for l in learners:
        filtered_success[l] = {length: np.array([result if length_ == length else 0 for result, length_ in zip(l.success, l.sent_len)]) for length in set(l.sent_len)}
        length_filtered[l] = {length: np.array([1 if length_ == length else 0 for result, length_ in zip(l.success, l.sent_len)]) for length in set(l.sent_len)}
    count = {}
    successes = {}
    for i in learners:
        for leng in filtered_success[learners[0]]:
            if leng not in count:
                count[leng]=0*filtered_success[learners[0]][leng]
            count[leng]+=length_filtered[i][leng]
            if leng not in successes:
                successes[leng] = 0*filtered_success[learners[0]][leng]
            successes[leng] += filtered_success[i][leng]
            


    successes_norm = {}
    for leng in successes:
        successes_norm[leng] = successes[leng]/count[leng]

        
    
    return successes_norm

def plot_success_norm(learners):
    successes_norm = success_by_length(learners)
    for key in successes_norm:
        plt.plot(successes_norm[key], '.', markersize=1,label=str(key))
    plt.legend()
    plt.show()
    
def plot_learning_curve(learnersC,learnersN,RWlearnersC,RWlearnersN):
    # Process the result   
    colors = ['m','m','m','m']
    ll = [learnersC,learnersN,RWlearnersC,RWlearnersN]
    lab = ['Q-learner, cont','Q-learner, next','RW Q-learner, cont', 'RW Q-learner, next']
    for i in [2]:
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
    plt.savefig('LearningCurveMD.pdf')
    plt.show()
    
    
def plot_learning_curve_length(learners):
    success, sent_len = get_success_and_length(learners)
    trial_vec = range(learners[0].n_trials+1)
    x_data = np.linspace(0,len(success),len(success))
    y_data = success
    popt,pcov = scipy.optimize.curve_fit(logistic,x_data,y_data,maxfev=10000)
    print('Optimal parameters')
    print(popt)
    print('learning time')
    print(2*popt[-1])
    print(np.diag(pcov)[-1])
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2,
                                    figsize=(12, 6))
    ax0.plot(x_data,logistic(x_data,*popt),'k:', label='_nolegend_')
    ax0.scatter(trial_vec,success,s = 2,c='m',label = 'RW Q-learner, cont')
    ax0.set_xlabel('Number of trials')
    ax0.set_ylabel('Fraction of correct responses')
    ax0.legend()
    
    successes_norm = success_by_length(learners)
    for key in successes_norm:
        ax1.plot(successes_norm[key], '.', markersize=1,label='len = '+str(key))
    ax1.legend()
    ax1.set_xlabel('Number of trials')
    plt.savefig('LearningCurveMD.pdf')
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
Learner.beta = 1.
Learner.positive_reinforcement = 25.
Learner.negative_reinforcement = -10.

RWLearner.alpha = 0.1
RWLearner.beta = 1.
RWLearner.positive_reinforcement = 25.
RWLearner.negative_reinforcement = -10.








#############################################################
#
#       Yang and Piantadosi grammar (level of reduction) with mono and ditransitive verbs
#
#############################################################

number_of_verbs = 1
number_of_nouns = 1
number_of_adj = 1
number_of_relpron = 1
number_of_det = 1
number_of_prep = 1
number_of_monotransitive_verbs = 1
number_of_ditransitive_verbs = 1

verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
det = ['d' + str(i) for i in range(1, number_of_det+1)]
prep = ['p' + str(i) for i in range(1, number_of_prep+1)]
monotransitive_verbs = ['mv' + str(i) for i in range(1, number_of_monotransitive_verbs+1)]
ditransitive_verbs = ['dv' + str(i) for i in range(1, number_of_ditransitive_verbs+1)]



terminalsYP = flatten([monotransitive_verbs,ditransitive_verbs,nouns,adjs,relpron,det,prep])
non_terminalsYP = ['S', 'N','NP','VP','V','rel','MV','DV','AP','PP','A','D','P']



production_rulesYP = {
    'S': [['NP', 'VP']],
    'NP': [['N'],['D','N']],#['D','AP','N'],['N','PP']],
    'VP': [['MV','NP'],['MV','NP','rel','MV','NP'],['DV','NP','NP'],['DV','NP','rel','MV','NP','NP']],
    'AP': [['A'],['A','A' ] ],
    'PP': [['P','N']],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)],
    'A': [['a' + str(i)] for i in range(1, number_of_adj+1)],
    'D': [['d' + str(i)] for i in range(1, number_of_det+1)],
    'P': [['p' + str(i)] for i in range(1, number_of_prep+1)],
    'rel': [['r' + str(i)] for i in range(1, number_of_relpron+1)],
    'MV': [['mv' + str(i)] for i in range(1, number_of_monotransitive_verbs+1)],
    'DV': [['dv' + str(i)] for i in range(1, number_of_ditransitive_verbs+1)]
}

weightsYP = {
    'S': [1.0],
    'NP': [.5,.5],
    'VP': [.25,.25,.25,.25],
    'AP': [.5,.5 ],
    'PP': [1.0],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)],
    'A': [1/number_of_adj for i in range(1, number_of_adj+1)],
    'D': [1/number_of_det for i in range(1, number_of_det+1)],
    'P': [1/number_of_prep for i in range(1, number_of_prep+1)],
    'rel': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
    'MV': [1/number_of_monotransitive_verbs for i in range(1, number_of_monotransitive_verbs+1)],
    'DV': [1/number_of_ditransitive_verbs for i in range(1, number_of_ditransitive_verbs+1)]
    }

cfgYPredMD = ProbabilisticGrammar(terminalsYP, non_terminalsYP, production_rulesYP,weightsYP)

#############################################################
#
#       Simple grammar with mono and ditransitive verbs
#
#############################################################

number_of_verbs = 1
number_of_nouns = 5
number_of_adj = 1
number_of_relpron = 1
number_of_det = 1
number_of_prep = 1
number_of_monotransitive_verbs = 1
number_of_ditransitive_verbs = 1

verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
det = ['d' + str(i) for i in range(1, number_of_det+1)]
prep = ['p' + str(i) for i in range(1, number_of_prep+1)]
monotransitive_verbs = ['mv' + str(i) for i in range(1, number_of_monotransitive_verbs+1)]
ditransitive_verbs = ['dv' + str(i) for i in range(1, number_of_ditransitive_verbs+1)]



terminalsYP = flatten([monotransitive_verbs,ditransitive_verbs,nouns,adjs,relpron,det,prep])
non_terminalsYP = ['S', 'N','NP','VP','V','rel','MV','DV','NPV','AP','PP','R','A','D','P']



production_rulesYP = {
    'S': [['NP', 'VP']],
    'NP': [['N']],#,['D','N'],['D','AP','N'],['N','PP']],
    'VP': [['MV','NP'],['DV','NP','NP']],
    'AP': [['A'],['A','A' ] ],
    'PP': [['P','N']],
    'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
    'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)],
    'A': [['a' + str(i)] for i in range(1, number_of_adj+1)],
    'D': [['d' + str(i)] for i in range(1, number_of_det+1)],
    'P': [['p' + str(i)] for i in range(1, number_of_prep+1)],
    'rel': [['r' + str(i)] for i in range(1, number_of_relpron+1)],
    'MV': [['mv' + str(i)] for i in range(1, number_of_monotransitive_verbs+1)],
    'DV': [['dv' + str(i)] for i in range(1, number_of_ditransitive_verbs+1)]
}

weightsYP = {
    'S': [1.0],
    'NP': [1.0],
    'VP': [.5,.5],
    'AP': [.75,.25 ],
    'PP': [1.0],
    'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
    'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)],
    'A': [1/number_of_adj for i in range(1, number_of_adj+1)],
    'D': [1/number_of_det for i in range(1, number_of_det+1)],
    'P': [1/number_of_prep for i in range(1, number_of_prep+1)],
    'rel': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
    'MV': [1/number_of_monotransitive_verbs for i in range(1, number_of_monotransitive_verbs+1)],
    'DV': [1/number_of_ditransitive_verbs for i in range(1, number_of_ditransitive_verbs+1)]
    }

cfgNVNMD = ProbabilisticGrammar(terminalsYP, non_terminalsYP, production_rulesYP,weightsYP)
#############################################################
#
#       Initializing the learners
#
#############################################################



# number of simulations
n_sim = 200
n_trials = 8000

#############################################################
#
#   Creating the stimuli stream
#
#############################################################
print('Creating the stimuli stream')

# Create stimuli stream
stimuli_stream = Raw_input(10*n_trials,cfgNVNMD)

print('Initializing learners')



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


        
# # Run the simulation in parallel
print('RW Q-learning with continuous condition')
with cf.ThreadPoolExecutor() as executor:
    print("Number of worker threads:", executor._max_workers)
    results = [executor.submit(run_simulation, l) for l in RWlearnersC]
    
    # Iterate over the results as they become available
    for future in cf.as_completed(results):
        result = future.result()
        

    
#############################################################
#
#       Postprocessing
#
#############################################################
print('Postprocessing')


#plot_learning_curve(RWlearnersC,RWlearnersC,RWlearnersC,RWlearnersC)
plot_learning_curve_length(RWlearnersC)


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

