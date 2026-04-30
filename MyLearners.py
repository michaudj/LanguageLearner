# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:26:00 2023

@author: jmd01
"""

#import json
from copy import deepcopy
import re
import numpy as np
import random
#from Raw_input import ProbabilisticGrammar

from itertools import accumulate
import pandas as pd

import sys
sys.setrecursionlimit(1500)



def modify_element_at_depth(nested_list, depth, new_value):
    for i in range(depth-1):
        nested_list = nested_list[-1]
    nested_list[-1] = [nested_list[-1],new_value]
    
def change_element_at_depth(nested_list, depth, new_value):
    for i in range(depth-1):
        nested_list = nested_list[-1]
    nested_list[-1] = new_value

    
def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def add_weights(b_values1,b_values2):
    a = b_values1[:]
    b = b_values2[:]
    l = sorted((a, b), key=len)
    c = l[1].copy()
    c[:len(l[0])] += l[0]
    return c



class Stimuli():
    
    def __init__(self,substantial,t='',v=0):
        self.content = substantial
        self.type = t
        self.value = v
        
    def __repr__(self):
        return str(self.content)# + ', t:'+str(self.type)+', v:'+str(self.value)+')'
    
    def longstr(self):
        return '('+str(self.content) + ', t:'+str(self.type)+', v:'+str(self.value)+')'
   

        
class SChunk():

    def __init__(self, structure):
        self.structure = structure
        self.depth = self.get_depth()
        
    def __repr__(self):
        return str(self.structure)
    
    def __hash__(self):
        return hash(frozenset((self.structure)))
    
    def get_s1(self):
        return SChunk(self.structure[0])
    
    def get_s2(self):
        return SChunk(self.structure[1])
    
    def __len__(self):
        return len(self.remove_structure())
    
    
    def get_right_subchunks(self, depth):
        right_subchunks = []
        nested_list = deepcopy(self.structure)
        for d in range(depth):
            nested_list = nested_list[-1]
            right_subchunks.append(SChunk(nested_list))
        return right_subchunks
    
    def chunk_at_depth(self, other, depth=0):
        nested_list = deepcopy(self.structure)
        if depth == 0:
            struct = [nested_list,other.structure]
            return SChunk(struct)
        else:
            modify_element_at_depth(nested_list, depth, other.structure)
            return SChunk(nested_list)
    
    def get_depth(self):
        st = str(self.structure)
        match = re.search("]*$",st)
        return len(match.group(0))
    
    def remove_structure(self):
        if type(self.structure) is not list:
            return [self.structure]
        else:
            return flatten(self.structure)
        

####################################################
####################################################
####################################################
####################################################
####################################################


class Learner():
    
    alpha = 0.2
    beta = 1.
    positive_reinforcement = 5.
    negative_reinforcement = -1.
    initial_value_chunking = -1.
    initial_value_border = 1.
    
    ID = 0
    
    def __init__(self, n_trials = 14, border = 'next'):
        self.ID = Learner.ID + 1
        Learner.ID +=1
        self.border_type = border # or 'default'
        self.n_trials = n_trials
        self.n_reinf = 0
        self.n_test = 0
        self.success = []
        self.test_success = []
        self.sent_len = []
        self.sentences = []
        self.sent_dict = dict()

        
        self.behaviour_repertoire = {} # dictionary of where the keys are couples of chunks and the value a list of behavioural values
        self.couple_str_to_couple = dict()
        self.events = [] # encodes the current list of couples ((chunk,chunk), behaviour) to reinforce
        self.stimuli = []
        self.decisions = []
        
        self.border_before = True
        self.border_within = False
        
        self.final_index = 0
        
        # for grammar extraction
        self.terminals = set()
        self.non_terminals = set()
        self.rules = dict()
        self.weights = dict()
        self.grammar = None
        
        
    def __repr__(self):
        return 'Learner ' + str(self.ID)
    
    def update_sent_dict(self,s1):
        unparsed_s1 = str(s1.remove_structure())
        parsing_s1 = str(s1)
        if unparsed_s1 not in self.sent_dict:
            self.sent_dict[unparsed_s1]={parsing_s1:1}
        else:
            if parsing_s1 not in self.sent_dict[unparsed_s1]:
                self.sent_dict[unparsed_s1][parsing_s1] = 1
            else:
                self.sent_dict[unparsed_s1][parsing_s1] += 1

            
    def update_repertoire(self,couple): # couple must be a couple of SChunks
        #print('call of update repertoire')
        substantial_couple = (str(couple[0]),str(couple[1]))
        if substantial_couple not in self.behaviour_repertoire:
            values = [Learner.initial_value_border]
            values += [Learner.initial_value_chunking for i in range(couple[0].get_depth()+1)]
            self.behaviour_repertoire[substantial_couple] =np.array(values)# np.array([Learner.initial_value_border] + [Learner.initial_value_chunking for i in range(couple[0].depth+1)])
        if substantial_couple not in self.couple_str_to_couple:
            self.couple_str_to_couple[substantial_couple] = couple

                
    def get_sub_couples(self, couple):
        sub_pairs = []
        for s in couple[0].get_right_subchunks(couple[0].get_depth()):

            sub_pairs.append((s,couple[1]))
            self.update_repertoire((s,couple[1]))
        return sub_pairs
    

    def learn(self,stimuli_stream):
        # initialize stimuli
        s1 = SChunk(stimuli_stream.stimuli[0])
        s2_index = 1
        #for t in range(self.n_trials):
        while self.n_reinf <= self.n_trials:
            s1, s2_index = self.respond(stimuli_stream, s1, s2_index)
            # under condition save snapshot to file
        self.final_index = s2_index
        
    def learn_n(self,stimuli_stream,n_trials):
        self.n_reinf = 0
        # initialize stimuli
        s1 = SChunk(stimuli_stream.stimuli[0])
        s2_index = 1
        #for t in range(self.n_trials):
        while self.n_reinf <= n_trials:
            s1, s2_index = self.respond(stimuli_stream, s1, s2_index)
            # under condition save snapshot to file
        self.final_index = s2_index
        
    def test(self,stimuli_stream,n_sent):
        self.test_success = []
        self.sentences = []
        self.sent_dict = dict()
        s1 = SChunk(stimuli_stream.stimuli[0])
        s2_index = 1
        for i in range(int(len(stimuli_stream.stimuli)/2)+1):
            s1, s2_index = self.respond_without_learning(stimuli_stream, s1, s2_index)
        print(len(self.sentences))
        
    def extract_test_sentences(self):
        length = []
        for s in self.sentences:
            length.append(len(s.remove_structure()))
        df = pd.DataFrame(zip(self.sentences,length), columns = ['parsed sentences','length'])
        return df
            

    def learn_with_snapshot(self,stimuli_stream,test_stimuli_stream,n_sent,filename,snap_times,threshold):
        # initialize stimuli
        s1 = SChunk(stimuli_stream.stimuli[0])
        s2_index = 1
        #for t in range(self.n_trials):
        done_snap = []
        with pd.ExcelWriter(filename) as f:
            dff = pd.DataFrame([(self.alpha, self.beta,self.positive_reinforcement,self.negative_reinforcement,self.initial_value_border,self.initial_value_chunking)],columns=['alpha','beta','positive reinforcement','negative reinforcement','initial value border','initial value chunking'])
            dff.to_excel(f,sheet_name='parameters')
            while self.n_reinf <= self.n_trials:
                s1, s2_index = self.respond(stimuli_stream, s1, s2_index)
                if self.n_reinf in snap_times and self.n_reinf not in done_snap:
                    done_snap.append(self.n_reinf)
                    print('begin test')
                    print(self.n_reinf)
                    self.test(test_stimuli_stream,n_sent)
                    dftst = self.extract_test_sentences()
                    dftst.to_excel(f,sheet_name=str(self.n_reinf)+'test')
                    print('End test')
                    df = self.extract_grammatical_information(threshold)
                    df.to_excel(f,sheet_name=str(self.n_reinf))
                    df = self.extract_grammatical_information2()
                    df.to_excel(f,sheet_name=str(self.n_reinf)+'full')
                    df = self.extract_grammatical_information_tilde()
                    df.to_excel(f,sheet_name=str(self.n_reinf)+'tilde')
    
    def respond_without_learning(self,stimuli_stream,s1,s2_index):
        # get the s2 stimuli and make it a chunk
        try:
            s2 = SChunk(stimuli_stream.stimuli[s2_index])
        except IndexError:
            sys.exit("Index doesn't exist. End of input reached before learning is finished.")
            

        response = self.choose_behaviour2((s1,s2))
                
        if response == 0: # boundary placement
            #print('Border')
            # increment the number of reinforcement events by 1
            self.n_test += 1 
            # check if border is correctly placed
            is_border = stimuli_stream.border_before[s2_index]

            if is_border and not self.border_within and self.border_before:
                #print('Good Unit')
                # perform positive reinforcement
                # Store sentence (not cognitively plausible but used for grammar extraction)
                
                self.sentences.append(s1)
                self.update_sent_dict(s1)
                # Perform reinforcement
                #self.reinforce(reinforcement = 'positive')                    
                # update the success list
                self.test_success.append(1)
                # for postprocessing, store length of the sentence
                # self.sent_len.append(stimuli_stream.length_current_sent(s2_index - 1))
            else:
                #print('Bad Unit')
                # perform negative reinforcement
                #self.reinforce(reinforcement = 'negative')
                # update the success list
                self.test_success.append(0)
                #self.sent_len.append(stimuli_stream.length_current_sent(s2_index))
            # Next beginning of sentence becomes
            if self.border_type == 'next':
                new_s1,s2_index = stimuli_stream.next_beginning_sent(s2_index)
                new_s1 = SChunk(new_s1)
            else:
                self.border_before = stimuli_stream.border_before[s2_index]
                new_s1,s2_index = s2, s2_index + 1

            self.border_within = False
              
        else: # some type of chunking occurs
            # Check if there was a border
            if not self.border_within:
                self.border_within = stimuli_stream.border_before[s2_index]
            
            # Perform chunking at correct level
            new_s1 = s1.chunk_at_depth(s2,depth=s1.get_depth()+1-response) 
            s2_index+=1      
        return new_s1, s2_index
    
    
    def respond(self,stimuli_stream,s1,s2_index):
        # get the s2 stimuli and make it a chunk
        try:
            s2 = SChunk(stimuli_stream.stimuli[s2_index])
        except IndexError:
            sys.exit("Index doesn't exist. End of input reached before learning is finished.")
            
        #s2 = SChunk(stimuli_stream.stimuli[s2_index])

        response = self.choose_behaviour2((s1,s2))

        self.events.append(((s1,s2),response))
        
        
                
        if response == 0: # boundary placement
            #print('Border')
            # increment the number of reinforcement events by 1
            self.n_reinf += 1 
            # check if border is correctly placed
            is_border = stimuli_stream.border_before[s2_index]

            if is_border and not self.border_within and self.border_before:
                #print('Good Unit')
                # perform positive reinforcement
                # Store sentence (not cognitively plausible but used for grammar extraction)
                #if self.n_reinf > 60000:
                #    self.sentences.add(str(s1))
                #    self.update_sent_dict(s1)
                # Perform reinforcement
                self.reinforce(reinforcement = 'positive')                    
                # update the success list
                self.success.append(1)
                # for postprocessing, store length of the sentence
                self.sent_len.append(stimuli_stream.length_current_sent(s2_index - 1))
            else:
                #print('Bad Unit')
                # perform negative reinforcement
                self.reinforce(reinforcement = 'negative')
                # update the success list
                self.success.append(0)
                self.sent_len.append(stimuli_stream.length_current_sent(s2_index))
            # Next beginning of sentence becomes
            if self.border_type == 'next':
                new_s1,s2_index = stimuli_stream.next_beginning_sent(s2_index)
                new_s1 = SChunk(new_s1)
            else:
                self.border_before = stimuli_stream.border_before[s2_index]
                new_s1,s2_index = s2, s2_index + 1

            self.border_within = False
              
        else: # some type of chunking occurs
            # Check if there was a border
            if not self.border_within:
                self.border_within = stimuli_stream.border_before[s2_index]
            
            # Perform chunking at correct level
            new_s1 = s1.chunk_at_depth(s2,depth=s1.get_depth()+1-response) 
            s2_index+=1      
        return new_s1, s2_index
    
    def choose_behaviour(self,couple):
        substantial_couple = (str(couple[0]),str(couple[1]))
        self.update_repertoire(couple)
        b_range = len(self.behaviour_repertoire[substantial_couple])
        options = [i for i in range(b_range)]
        z = deepcopy(self.behaviour_repertoire[substantial_couple])
        subpairs = self.get_sub_couples(couple)
        
        norm_vec = np.array([b_range - 1]+[i for i in range(b_range-1,0,-1)])
        # Accumulate support from subchunks
        for pair in subpairs:
            substantial_pair = (str(pair[0]),str(pair[1]))
            lenp = len(self.behaviour_repertoire[substantial_pair])
            z[:lenp] += self.behaviour_repertoire[substantial_pair]
        # Take the average
        z /= norm_vec
        weights = np.exp(Learner.beta * z)

        response = random.choices(options,weights/np.sum(weights))
        return response[0]  
    
    def choose_behaviour2(self,couple):
        self.update_repertoire(couple)
        substantial_couple = (str(couple[0]),str(couple[1]))
        b_range = len(self.behaviour_repertoire[substantial_couple])
        z = self.Q_tilde(couple)
        weights = np.exp(Learner.beta * z)
        options = [i for i in range(b_range)]
        response = random.choices(options,weights/np.sum(weights))
        return response[0]  
    
    def Q_tilde(self,couple):
        substantial_couple = (str(couple[0]),str(couple[1]))
        b_range = len(self.behaviour_repertoire[substantial_couple])
        z = deepcopy(self.behaviour_repertoire[substantial_couple])
        subpairs = self.get_sub_couples(couple)
        
        norm_vec = np.array([b_range - 1]+[i for i in range(b_range-1,0,-1)])
        # Accumulate support from subchunks
        for pair in subpairs:
            substantial_pair = (str(pair[0]),str(pair[1]))
            lenp = len(self.behaviour_repertoire[substantial_pair])
            z[:lenp] += self.behaviour_repertoire[substantial_pair]
        # Take the average
        z /= norm_vec
        return z 

    def reinforce(self, reinforcement = 'positive'):
        #print('call of reinforce')
        # for each events reinforce behaviour associated to chunk
        if reinforcement == 'positive':
            u = Learner.positive_reinforcement
        elif reinforcement == 'negative':
            u = Learner.negative_reinforcement
        #print(self.events)
        

        
        for couple,r in self.events:
            #substantial_couple= (str(couple[0]),str(couple[1]))
            #print('reinforcement')
            #Q = self.behaviour_repertoire[substantial_couple][r]
            subevents=[(couple,r)]
            subpairs = self.get_sub_couples(couple)
            for pair in subpairs:
                substantial_pair = (str(pair[0]),str(pair[1]))
                self.update_repertoire(pair)
                if r < len(self.behaviour_repertoire[substantial_pair]):
                    subevents.append((pair,r))
                    # Q += self.behaviour_repertoire[substantial_pair][r]
                    #print(subevents)
            for p,rr in subevents:
                substantial_p = (str(p[0]),str(p[1]))
                self.behaviour_repertoire[substantial_p][rr] += Learner.alpha * (u - self.behaviour_repertoire[substantial_p][rr])

        # Clear working memory
        self.events = []
        
    def extract_grammatical_information(self,threshold):
        grammar = list()
        for key,value in self.behaviour_repertoire.items():
            z = self.Q_tilde(self.couple_str_to_couple[key])
            weights = np.exp(Learner.beta * z)
            weights /= np.sum(weights)
            length = len(self.couple_str_to_couple[key][0].remove_structure())
            if max(value)>threshold and length < 4:
                grammar.append((key[0],key[1],length,list(value).index(max(value)),max(value),list(z).index(max(z)),max(z),max(weights)))
        df = pd.DataFrame(grammar, columns = ['s1','s2','Length s1','Index','Q Value','Index','Q bar Value','Proba'])
        return df
    
    def extract_grammatical_information2(self):
        grammar = list()
        for key,value in self.behaviour_repertoire.items():
            z = self.Q_tilde(self.couple_str_to_couple[key])
            weights = np.exp(Learner.beta * z)
            weights /= np.sum(weights)
            length = len(self.couple_str_to_couple[key][0].remove_structure())
            if length < 4:
                grammar.append((key[0],key[1],length,list(value).index(max(value)),max(value),list(z).index(max(z)),max(z),max(weights)))
        df = pd.DataFrame(grammar, columns = ['s1','s2','Length s1','Index','Q Value','Index','Q bar Value','Proba'])
        return df
    
    def extract_grammatical_information_tilde(self):
        grammar = list()
        for key,value in self.behaviour_repertoire.items():
            z = self.Q_tilde(self.couple_str_to_couple[key])
            weights = np.exp(Learner.beta * z)
            weights /= np.sum(weights)
            length = len(self.couple_str_to_couple[key][0].remove_structure())
            #grammar.append((key[0],key[1],list(weights).index(max(weights)),max(weights),list(value).index(max(value)),max(value)))
            if length < 4:
                grammar.append((key[0],key[1],length,value,z,weights))
        df = pd.DataFrame(grammar, columns = ['s1','s2','length s1','Q','Q tilde','proba'])
        return df


class RWLearner(Learner):
    
    def __init__(self, n_trials = 14, border = 'next'):
        super().__init__(n_trials =n_trials, border = border)
        
    def choose_behaviour(self,couple):
        substantial_couple = (str(couple[0]),str(couple[1]))
        self.update_repertoire(couple)
        b_range = len(self.behaviour_repertoire[substantial_couple])
        options = [i for i in range(b_range)]
        z = deepcopy(self.behaviour_repertoire[substantial_couple])
        subpairs = self.get_sub_couples(couple)
        
        norm_vec = np.array([b_range - 1]+[i for i in range(b_range-1,0,-1)])
        # Accumulate support from subchunks
        for pair in subpairs:
            substantial_pair = (str(pair[0]),str(pair[1]))
            lenp = len(self.behaviour_repertoire[substantial_pair])
            z[:lenp] += self.behaviour_repertoire[substantial_pair]
        # Take the average
        z /= norm_vec
        weights = np.exp(RWLearner.beta * z)

        response = random.choices(options,weights/np.sum(weights))
        return response[0] 
        
    def reinforce(self, reinforcement = 'positive'):
        #print('call of reinforce')
        # for each events reinforce behaviour associated to chunk
        if reinforcement == 'positive':
            u = Learner.positive_reinforcement
        elif reinforcement == 'negative':
            u = Learner.negative_reinforcement
        #print(self.events)
        

        
        for couple,r in self.events:
            substantial_couple= (str(couple[0]),str(couple[1]))
            #print('reinforcement')
            Q = self.behaviour_repertoire[substantial_couple][r]
            subevents=[(couple,r)]
            subpairs = self.get_sub_couples(couple)
            for pair in subpairs:
                substantial_pair = (str(pair[0]),str(pair[1]))
                self.update_repertoire(pair)
                if r < len(self.behaviour_repertoire[substantial_pair]):
                    subevents.append((pair,r))
                    Q += self.behaviour_repertoire[substantial_pair][r]
                    #print(subevents)
            for p,rr in subevents:
                substantial_p = (str(p[0]),str(p[1]))
                self.behaviour_repertoire[substantial_p][rr] += RWLearner.alpha * (u - Q)

        # Clear working memory
        self.events = []    
    
