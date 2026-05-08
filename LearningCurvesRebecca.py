# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:34:17 2026

@author: jemi6917
"""



from MyLearners import Learner,RWLearner
from Raw_input import Raw_input, ProbabilisticGrammar
import numpy as np
import random
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
    l.learn_n(stimuli_streams[0],n_trials)
    l.learn_n(stimuli_streams[1],n_trials)
    return l

def run_simulation_chrono(l):
    for i in range(len(stimuli_streams)):
        print(i)
        l.learn_n(stimuli_streams[i],n_trials)
    return l

def run_simulation_anti_chrono(l):
    n= len(stimuli_streams)
    for i in range(n):
        print(i)
        l.learn_n(stimuli_streams[n-i-1],n_trials)
    return l

def run_simulation_random(l):
    n= len(stimuli_streams)
    shuffled = random.sample(stimuli_streams,n)
    for i in range(n):
        print(i)
        l.learn_n(shuffled[i],n_trials)
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


def moving_average(x, window):
    return np.convolve(x, np.ones(window)/window, mode='valid')

    
def plot_learning_curve(learnersC,learnersN,RWlearnersC,RWlearnersN):
    # Process the result   
    colors = ['b','g','m','c']
    ll = [learnersC,learnersN,RWlearnersC,RWlearnersN]
    lab = ['Q-learner, cont','Q-learner, next','RW Q-learner, cont', 'RW Q-learner, next']
    for i in range(4):
        success, sent_len = get_success_and_length(ll[i])
        
        
        window = 200  # number of trials in the moving window

        ma_success = moving_average(success, window)
        ma_trials = np.arange(len(ma_success)) + window // 2

        plt.plot(ma_trials, ma_success, color=colors[i], label=lab[i])

    
        trial_vec = range(len(success))#range(ll[i][0].n_trials+1)
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
        #plt.scatter(trial_vec,success,s = 2,c=colors[i],label = lab[i])
        plt.xlabel('Number of trials')
        plt.ylabel('Fraction of correct responses')
    #plt.colorbar()
    plt.legend()
    plt.savefig('LearningCurves.pdf')
    plt.show()
    
    
def plot_learning_curve2(chrono,anti,random):
    # Process the result   
    colors = ['b','g','m']
    ll = [chrono, anti, random]
    lab = ['Chronological','Reverse','Random']
    for i in range(3):
        success, sent_len = get_success_and_length(ll[i])
        
        
        window = 100  # number of trials in the moving window

        ma_success = moving_average(success, window)
        ma_trials = np.arange(len(ma_success)) + window // 2

        plt.plot(ma_trials, ma_success, color=colors[i], label=lab[i])


        #plt.axvline(x = 2*popt[-1],color = 'k')
        # Plot the results
        #plt.scatter(trial_vec,success,s = 2,c=colors[i],label = lab[i])
        plt.xlabel('Number of trials')
        plt.ylabel('Fraction of correct responses')
        
        
        # Vertical lines every 5000 trials
        x_max = plt.gca().get_xlim()[1]
        for x in range(0, int(x_max) + 1, n_trials):
            plt.axvline(x=x, color='k', linestyle='--', alpha=0.3)

    #plt.colorbar()
    plt.legend()
    plt.savefig('LearningCurves_Comparison_RWQlearners100-550.pdf')
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
#       Initializing the learners
#
#############################################################

# number of simulations
n_sim = 10
n_trials = 450

#############################################################
#
#   Creating the stimuli stream
#
#############################################################
print('Creating the stimuli stream')

stimuli_streams = []
stimuli_streams.append(Raw_input.from_csv_with_copies('data/3mo_tagged.csv', n_copies=1, shuffle=True))
stimuli_streams.append(Raw_input.from_csv_with_copies('data/6mo_tagged.csv', n_copies=1, shuffle=True))
stimuli_streams.append(Raw_input.from_csv_with_copies('data/9mo_tagged.csv', n_copies=1, shuffle=True))
stimuli_streams.append(Raw_input.from_csv_with_copies('data/12mo_tagged.csv', n_copies=1, shuffle=True))
stimuli_streams.append(Raw_input.from_csv_with_copies('data/15mo_tagged.csv', n_copies=1, shuffle=True))
stimuli_streams.append(Raw_input.from_csv_with_copies('data/18mo_tagged.csv', n_copies=1, shuffle=True))
stimuli_streams.append(Raw_input.from_csv_with_copies('data/21mo_tagged.csv', n_copies=1, shuffle=True))
stimuli_streams.append(Raw_input.from_csv_with_copies('data/24mo_tagged.csv', n_copies=1, shuffle=True))
stimuli_streams.append(Raw_input.from_csv_with_copies('data/27mo_tagged.csv', n_copies=1, shuffle=True))
stimuli_streams.append(Raw_input.from_csv_with_copies('data/30mo_tagged.csv', n_copies=1, shuffle=True))
stimuli_streams.append(Raw_input.from_csv_with_copies('data/33mo_tagged.csv', n_copies=1, shuffle=True))
stimuli_streams.append(Raw_input.from_csv_with_copies('data/36mo_tagged.csv', n_copies=1, shuffle=True))



print('Initializing learners')

# Select type of chunking mechanism
#typ = 'right'
typ = 'flexible'
border = 'nxt'

# Create as many learners as number of simulations.
antilearnersN = [RWLearner(n_trials = n_trials, border = 'cont') for i in range(n_sim)]
chronolearnersN = [RWLearner(n_trials = n_trials, border = 'cont') for i in range(n_sim)]
randlearnersN = [RWLearner(n_trials = n_trials, border = 'cont') for i in range(n_sim)]
#RWlearnersN = [RWLearner(n_trials = n_trials, border = 'next') for i in range(n_sim)]

#learner = RWLearner(n_trials = n_trials, border = border)
#learner.learn_with_snapshot(stimuli_stream, 'test.xlsx', [1000,2000,3000,4000], 5)


#############################################################
#
#       Running the simulation in parallel
#
#############################################################    
print('Running the simulation in parallel')

# print('Q-learning with continuous border')
# # # Run the simulation in parallel
# with cf.ThreadPoolExecutor() as executor:
#     print("Number of worker threads:", executor._max_workers)
#     results = [executor.submit(run_simulation_chrono, l) for l in learnersC]
    
#     # Iterate over the results as they become available
#     for future in cf.as_completed(results):
#         result = future.result()
#         # Combine the result with other results as necessary
        
# # Run the simulation in parallel
print('Q-learning with next sentence condition: chronological')
with cf.ThreadPoolExecutor() as executor:
    print("Number of worker threads:", executor._max_workers)
    results = [executor.submit(run_simulation_chrono, l) for l in chronolearnersN]
    
    # Iterate over the results as they become available
    for future in cf.as_completed(results):
        result = future.result()
        
# # Run the simulation in parallel
print('Q-learning with next sentence condition: reverse chronological')
with cf.ThreadPoolExecutor() as executor:
    print("Number of worker threads:", executor._max_workers)
    results = [executor.submit(run_simulation_anti_chrono, l) for l in antilearnersN]
    
    # Iterate over the results as they become available
    for future in cf.as_completed(results):
        result = future.result()
        
# # Run the simulation in parallel
print('Q-learning with next sentence condition: random')
with cf.ThreadPoolExecutor() as executor:
    print("Number of worker threads:", executor._max_workers)
    results = [executor.submit(run_simulation_random, l) for l in randlearnersN]
    
    # Iterate over the results as they become available
    for future in cf.as_completed(results):
        result = future.result()
    
#############################################################
#
#       Postprocessing
#
#############################################################
print('Postprocessing')

plot_learning_curve2(chronolearnersN,antilearnersN,randlearnersN)
# plot_learning_curve(chronolearnersN,antilearnersN,chronolearnersN,antilearnersN)
#print(get_averaged_final_index(learnersC))
# print(get_averaged_final_index(learnersN))
#print(get_averaged_final_index(RWlearnersC))
# print(get_averaged_final_index(RWlearnersN))


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

