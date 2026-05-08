# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:49:31 2019

@author: jermi792
"""

import random
import csv
random.seed()

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


class ProbabilisticGrammar:
    def __init__(self, terminals, non_terminals, production_rules,weights):
        self.terminals = terminals
        self.non_terminals = non_terminals
        self.production_rules = production_rules
        self.weights = weights
        
    def __repr__(self):
        non_terminals_str = ', '.join(self.non_terminals)
        production_rules_str = '\n'.join([f"{k} -> {v}" for k, v in self.production_rules.items()])
        weights_str = '\n'.join([f"{k} -> {v}" for k, v in self.weights.items()])
        return f"terminals={self.terminals},\nnon_terminals=[{non_terminals_str}],\nproduction_rules=\n{production_rules_str},\nweights=\n{weights_str}"

    def __add__(self,other):
        new_terminals = list(set(self.terminals + other.terminals))
        new_non_terminals = list(set(self.non_terminals + other.non_terminals))

        new_production_rules = dict()
        new_production_rules.update(self.production_rules)
        new_production_rules.update(other.production_rules)

        new_weights = dict()
        new_weights.update(self.weights)
        new_weights.update(other.weights)

        return ProbabilisticGrammar(new_terminals, new_non_terminals, new_production_rules, new_weights)
    
    def generate_sentence(self, symbol):
        if symbol in self.terminals:
            return [symbol]
        
        rules = self.production_rules[symbol]
        weights = self.weights[symbol]
        
        chosen_rule = random.choices(rules, weights=weights)[0]
        
        return flatten([self.generate_sentence(s) for s in chosen_rule])




class Raw_input():
    
    
    def __init__(self, n_sentence,cfg):
        #self.number_nouns =
        self.stimuli = []
        self.border_before = []
        for i in range(n_sentence):
            sentence = cfg.generate_sentence('S')#self.generate_sentence(cfg,symbol)
            self.stimuli += sentence
            border_before = [False for i in range(len(sentence))]
            border_before[0] = True
            self.border_before += border_before
            
    @classmethod
    def from_csv(cls, filepath):
        obj = cls.__new__(cls)  # create instance without calling __init__
        obj.stimuli = []
        obj.border_before = []

        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not row:
                    continue
               
                # If sentence is stored as one string per row
                sentence = row[0].split()   # split into tokens
               
                obj.stimuli += sentence
                border_before = [False] * len(sentence)
                border_before[0] = True
                obj.border_before += border_before

        return obj

    @classmethod
    def from_csv_with_copies(cls, filepath, n_copies=1, shuffle=True, seed=None):
        obj = cls.__new__(cls)  # create instance without calling __init__
        obj.stimuli = []
        obj.border_before = []

        # --- Read sentences first (sentence-level structure) ---
        sentences = []

        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not row:
                    continue

                # sentence stored as one string per row
                sentence = row[0].split()
                if sentence and len(sentence)>2 and len(sentence)<15:
                    sentences.append(sentence)

        # --- Optional reproducibility ---
        if seed is not None:
            random.seed(seed)
            
        # --- Shuffle sentences (not tokens) ---
        if shuffle:
            random.shuffle(sentences)
            
        print(filepath)
        print('number of sentences:'+str(len(sentences)))
        unique_count = len(set(tuple(sent) for sent in sentences))
        print('unique sentences'+str(unique_count))

        # --- Duplicate dataset ---
        sentences = sentences * n_copies


        # --- Flatten into stimuli and border_before ---
        for sentence in sentences:
            obj.stimuli += sentence

            border_before = [False] * len(sentence)
            border_before[0] = True
            obj.border_before += border_before

        return obj

    
    
    # def generate_sentence(self,cfg, symbol):
    #     if symbol in cfg['terminals']:
    #         return [symbol]
    #     expansion = flatten(random.choices(cfg['production_rules'][symbol],cfg['weights'][symbol]))
    #     return flatten([self.generate_sentence(cfg, s) for s in expansion])
    
    def next_beginning_sent(self,index):
        for i in range(index, len(self.border_before)):
            if self.border_before[i] == True:
                return (self.stimuli[i],i+1)
        return None
    
    def length_current_sent(self,index):
        count_before = 0
        for i in range(index, -1, -1):
            if self.border_before[i] == False:
                count_before += 1
            else:
                break
        count_after = 0
        for i in range(index, len(self.border_before)):
            if self.border_before[i] == False:
                count_after += 1
            else:
                break
        return count_before+ count_after

                                
# definition of the grammar
# Vocabulary
# number_of_verbs = 2
# number_of_nouns = 2
# number_of_adj = 0
# number_of_relpron = 1
# number_of_det = 0

# verbs = ['v' + str(i) for i in range(1, number_of_verbs+1)]
# nouns = ['n' + str(i) for i in range(1, number_of_nouns+1)]
# adjs = ['a' + str(i) for i in range(1, number_of_adj+1)]
# relpron = ['r' + str(i) for i in range(1, number_of_relpron+1)]
# det = ['d' + str(i) for i in range(1, number_of_det+1)]

# terminals2 = flatten([verbs,nouns,adjs,relpron,det])
# non_terminals2 = ['S', 'N','NP','VP','V']

# # Grammatical rules
# production_rules2 = {
#     'S': [['N', 'VP'],['N','VP','RelCl']],
#     'VP': [['V','N'],['V']],
#     'RelCl': [['r' + str(i) , 'VP'] for i in range(1, number_of_relpron+1)],
#     'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
#     'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
# }

# weights = {
#     'S': [0.8, 0.2],
#     'VP': [0.8, 0.2],
#     'RelCl': [1/number_of_relpron for i in range(1, number_of_relpron+1)],
#     'N': [1/number_of_nouns for i in range(1, number_of_nouns+1)],
#     'V': [1/number_of_verbs for i in range(1, number_of_verbs+1)]    
#     }

# # Grammatical rules
# production_rules = {
#     'S': [['N', 'VP']],
#     'VP': [['V','N']],
#     'RelCl': [['r' + str(i) , 'VP'] for i in range(1, number_of_relpron+1)],
#     'N': [['n' + str(i)] for i in range(1, number_of_nouns+1)],
#     'V': [['v' + str(i)] for i in range(1, number_of_verbs+1)]
# }
# # Context free grammar
# cfg = {
#     'terminals': terminals2,
#     'non_terminals': non_terminals2,
#     'production_rules': production_rules2,
#     'weights': weights
# }


# stimuli_stream = Raw_input(20,cfg,'S')
# print(stimuli_stream.stimuli)