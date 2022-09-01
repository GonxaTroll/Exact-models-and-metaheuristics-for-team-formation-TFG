import pandas as pd
import itertools
import numpy as np
import math
import random
import Levenshtein
from ast import literal_eval
import sys
import time
sys.path.insert(0,"..")
from general_functions import *

class StudentGA:
    def __init__(self,students,scores_d,compulsory,reject,mingroup,maxgroup,dict_rest,scoring_type=1):
        self.students = students
        self.scores_d = scores_d
        self.compulsory = compulsory
        self.reject = reject
        self.mingroup = mingroup
        self.maxgroup = maxgroup
        self.dict_rest = dict_rest
        self.valid_masks = self.generate_valid_masks()
        self.distance_matrix = self.generate_distance_matrix()
        if scoring_type==1:
            self.score_function = belbin_score
        elif scoring_type==2:
            self.score_function = mbti_score
        
    def generate_random_solution(self):
        students = list(self.students)
        random.shuffle(students)
        random_mask = random.choice(self.valid_masks)
        return [students,random_mask]
    
    def generate_random_population(self,population_size=100):
        return [self.generate_random_solution() for i in range(population_size)]
    
    def save_random_population(self,filename,population_size=100):
        population = self.generate_random_population(population_size)
        with open(filename,"w") as file:
            for solution in population:
                file.write(str(solution)+"\n")
                
    def load_random_population(self,filename):
        with open(filename,"r") as file:
            return [literal_eval(line) for line in file]
     
    def generate_valid_masks(self):
        phenotypes = []
        n = len(self.students)
        lsizes = list(range(self.mingroup,self.maxgroup+1))
        combinations = itertools.combinations_with_replacement(lsizes,math.ceil(n/self.mingroup))
        for c in combinations:
            current_students = 0
            pheno = []
            for element in c:
                current_students+=element
                pheno.append(element)
                if current_students > n:
                    break
                elif current_students == n:
                    if pheno not in phenotypes:
                        phenotypes.append(pheno)
                    break
        valid_masks = []
        for pheno in phenotypes:
            mask = []
            for groupsize in pheno:
                for i in range(groupsize-1):
                    mask.append(0)
                mask.append(1)
            mask.pop()
            valid_masks.append(mask)
        return valid_masks
    
    def generate_distance_matrix(self):
        valid_str_masks = np.array(["".join(list(map(str,m))) for m in self.valid_masks])
        n = len(valid_str_masks)
        matrix = np.zeros((n,n))
        for mk1_pos in range(n):
            for mk2_pos in range(n):
                matrix[mk1_pos,mk2_pos] = Levenshtein.distance(valid_str_masks[mk1_pos],valid_str_masks[mk2_pos])
        return matrix
    
    def decode_solution(self,solution):
        students,mask = solution[0],solution[1]
        group = {students[0]}
        groups = []
        for i in range(len(mask)):
            if mask[i] == 1:
                groups.append(group)
                group = {students[i+1]}
            else:
                group.add(students[i+1])
        groups.append(group)
        return groups

    def encode_solution(self,groups):
        lengths = [len(x) for x in groups]
        sorted_lists = [(lens,students) for lens, students in sorted(zip(lengths,groups))] #preserve the phenotype order
        students = []
        for s in [x[1] for x in sorted_lists]:
            students += s
        mask = []
        for length in [x[0] for x in sorted_lists]:
            for i in range(length-1):
                mask.append(0)
            mask.append(1)
        mask.pop()
        return [students,mask]
    
    def get_fitness(self,solution):
        teams = self.decode_solution(solution)
        score = 0
        if self.validate_comp_reject_solution(teams) is False:
            return score
        for team in teams:
            score += self.score_function(self.scores_d,team, self.students)
        return score

    def randomised_mask_crossover(self,mask1,mask2):
        crossed_masks = single_point_crossover(mask1,mask2)
        new_masks = []
        for mask in crossed_masks:
            mask_str = "".join(list(map(str,mask)))
            distances = [Levenshtein.distance(mask_str,"".join(list(map(str,m)))) for m in self.valid_masks]
            closest_dist = min(distances)
            closest_masks = [self.valid_masks[i] for i in range(len(distances)) if distances[i]==closest_dist]
            closest_mask = random.choice(closest_masks)
            new_masks.append(closest_mask)
        return new_masks
    
    def mask_mutation(self,mask):
        sp = 1.5
        if len(self.valid_masks)==1:
            return self.valid_masks[0]
        distance_row = self.distance_matrix[self.valid_masks.index(mask),:]
          
        indices = [i for i in range(distance_row.size)]
        ranks = np.array([i for i in range(1,distance_row.size+1)])
        ref = sorted(zip(distance_row,indices),reverse=True) 
        sorted_indices = [x for _,x in ref] 
        scaled_ranks = 2-sp + (2*(sp-1)*(ranks-1)/(ranks.size-1))
        selection_probs = scaled_ranks/np.sum(scaled_ranks)
        selected_index = np.random.choice(sorted_indices,p=selection_probs)
        return self.valid_masks[selected_index]

    def randomised_mask_mutation(self,mask):
        if len(self.valid_masks)==1:
            return self.valid_masks[0]
        mutated_mask = bitwise_mutation(mask)
        mutated_mask_str = "".join(list(map(str,mutated_mask)))
        distances = [Levenshtein.distance(mutated_mask_str,"".join(list(map(str,m)))) for m in self.valid_masks]
        closest_dist = min(distances)
        closest_masks = [self.valid_masks[i] for i in range(len(distances)) if distances[i]==closest_dist]
        mutated_mask = random.choice(closest_masks)
        return mutated_mask
    
    def validate_comp_reject_solution(self,solution):
        """Checks if a solution is feasible"""
        for team in solution:
            for cteam in self.compulsory:
                if cteam.intersection(team) and cteam.intersection(team)!=cteam:
                    return False
            for rteam in self.reject:
                if rteam.intersection(team) and rteam.intersection(team)==rteam:
                    return False
        return True

    def ranked_wheel_selection(self,population,sp,parent_number):
        sorted_pop = sorted(population,key=self.get_fitness)
        sorted_indices = list(range(len(population)))
        ranks = np.array(list(range(1,len(population)+1)))
        scaled_ranks = 2-sp + (2*(sp-1)*(ranks-1)/(ranks.size-1))
        selection_probs = scaled_ranks/np.sum(scaled_ranks)
        selected_indices = np.random.choice(sorted_indices,size=parent_number,p=selection_probs)
        new_pop = [sorted_pop[x] for x in selected_indices]
        return new_pop

    def elitist_replacement(self,population, children):
        new_population = population + children
        new_population = sorted(new_population, key = lambda x:self.get_fitness(x),reverse=True)
        return new_population[0:len(population)]
    
    def execute(self,configuration=None):
        if configuration is None:
            configuration = ('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 50)

        final_variables = ["best_fitness", "iterations", "final_time", "final_population"]
        iteration_data = []
        ini_time = time.time()
        iterations = 0

        cross_name = configuration[0]
        p_cross = configuration[1]
        mask_cross = configuration[2]
        mut_name = configuration[3]
        p_mut = configuration[4]
        mask_mut = configuration[5]
        sel_name = configuration[6]
        rep_name = configuration[7]
        pop_size = configuration[8]

        if cross_name=="NWOX":
            crossover = NWOX
        elif cross_name=="CX":
            crossover = CX
        elif cross_name=="CX2":
            crossover = CX2_modded
        if mut_name=="SS":
            mutation = simple_swap
        elif mut_name=="RSM":
            mutation = RSM

        if mask_mut == "DivisionSelect":
            mask_mutation = self.mask_mutation
        elif mask_mut == "RandomisedMasking":
            mask_mutation = self.randomised_mask_mutation

        if "RWS" in sel_name:
            selection = self.ranked_wheel_selection
        if rep_name=="EL":
            replacement = self.elitist_replacement

        population = self.generate_random_population(population_size=pop_size)
        generated_children = len(population)//2

        fitnesses = [self.get_fitness(x) for x in population]
        best_fitness_individual_iteration = max(fitnesses)
        best_individual_iteration = population[fitnesses.index(best_fitness_individual_iteration)]
        iteration_data.append([best_fitness_individual_iteration,0,0,"pop"])
        best_last_fitness = best_fitness_individual_iteration
        repeated_iterations=0

        while iterations < 500:
            ini_iteration_time = time.time()
            parents = selection(population, 1.5, generated_children)

            random.shuffle(parents)
            final_children, new_children = [],[]
            for i in range(0, generated_children-1, 2):
                contador = 0
                provisional_children = []
                while len(new_children) < 2 or contador < 5:
                    parent1_perm = parents[i][0]
                    parent2_perm = parents[i+1][0]
                    keys = [parents[i][1], parents[i+1][1]]

                    r = random.random()
                    if r <= p_cross:
                    
                        children_perm = crossover(parent1_perm, parent2_perm)
                        random.shuffle(keys)
                    else:
                        children_perm = [parent1_perm, parent2_perm]

                    mutated_children = []
                    for i in range(len(children_perm)):
                        mutated_child_perm, mutated_key = children_perm[i],keys[i]
                        r = random.random()
                        if r <= p_mut:
                            mutated_child_perm = mutation(mutated_child_perm)
                            mutated_key = mask_mutation(mutated_key)
                        mutated_children.append([mutated_child_perm,mutated_key])

                    for child in mutated_children:
                        if self.validate_comp_reject_solution(self.decode_solution(child)):
                            new_children.append(child)
                        else:
                            provisional_children.append(child)
                    contador+=1

                if len(new_children)<2:
                    rest = random.choices(provisional_children,k=2-len(new_children))
                    for c in rest:
                        new_children.append(c)
                for c in new_children:
                    final_children.append(c)

            population=replacement(population, final_children)

            best_individual_iteration = population[0]
            best_fitness_individual_iteration = self.get_fitness(best_individual_iteration)
            iterations+=1
            end_iteration_time = time.time()-ini_iteration_time
            iteration_data.append([best_fitness_individual_iteration, iterations, end_iteration_time, "pop"])

            if best_fitness_individual_iteration <= best_last_fitness:
                repeated_iterations += 1
            else:
                repeated_iterations = 0
            best_last_fitness = best_fitness_individual_iteration
    
            if iterations==500 or repeated_iterations == 100: 
                final_time = time.time()-ini_time
                result_data = [[best_fitness_individual_iteration, iterations, final_time, population]]
                result_data = pd.DataFrame(result_data,columns=final_variables)
                iteration_data = pd.DataFrame(iteration_data,columns=final_variables)
                iteration_data.drop(columns=["final_population"],inplace=True)
                return result_data,iteration_data

## Other generic operators

## 1. Crossover Operators  
         
def aux_NWOX(parent1,parent2,i,j): # Non-Wrapping Ordered Crossover (auxiliary function)
    child = [parent1[x] if i<=x<=j else None for x in range(len(parent1))] # Copying the block of one parent
    search_ind = 0
    for e2 in parent2: # Adding the rest of the elements in the order of the other parent
        if e2 not in child:
            while search_ind < len(child) and child[search_ind] is not None:
                search_ind += 1
            child[search_ind] = e2
    return child

def NWOX(parent1,parent2): # Non-Wrapping Ordered Crossover
    i = random.randint(0,len(parent1)-1)
    j = random.randint(0,len(parent1)-1)
    if j < i:
        i,j = j,i
    child1 = aux_NWOX(parent1,parent2,i,j)
    child2 = aux_NWOX(parent2,parent1,i,j)
    return [child1,child2]

def aux_CX(parent1,parent2): # Cycle Crossover (auxiliary function)
    parents = [parent1,parent2]
    N = len(parent1)
    child= [-1 for x in range(N)]
    end_cycle = True
    while sum(parent1) != sum(child):
        if end_cycle:
            current_index = child.index(-1)
            end_cycle = False
        else:
            posibles = [parent1[current_index],parent2[current_index]]
            choice = random.choice(posibles)
            selected_index_parent = posibles.index(choice)
            if choice in child:
                for i,p in enumerate(posibles):
                    if i != selected_index_parent:
                        if p in child:
                            end_cycle=True
                        else:
                            choice=p
                            selected_index_parent = i
                        break
            other_index_parent = abs(1-selected_index_parent)
            if choice not in child:
                child[current_index] = choice
                current_index = parents[other_index_parent].index(choice)
    return child

def CX(parent1,parent2): # Cycle Crossover
    return [aux_CX(parent1,parent2),aux_CX(parent2,parent1)]

def CX2_modded(parent1,parent2): # Cycle Crossover 2 corrected
    child1,child2 = [],[]
    original_parent1 = list(parent1)
    while len(child1)!=len(original_parent1):
        it = 0
        while True:
            if it==0:
                child1.append(parent2[0])
                it+=1
            else:
                child1.append(new_valueoff1)
            ref1p1 = parent1.index(child1[-1])
            ref2p1 = parent1.index(parent2[ref1p1])
            new_valueoff2 = parent2[ref2p1]
            child2.append(new_valueoff2)
            new_valueoff1 = parent2[parent1.index(new_valueoff2)]
            if new_valueoff1 in child1:
                break
        common = set(child1).intersection(child2)
        if len(common) != len(child1):
            child1 = child1 + [x for x in parent2 if x not in child1]
            child2 = child2 + [x for x in parent1 if x not in child2]
            break
        else:
            parent1 = [x for x in parent1 if x not in common]
            parent2 = [x for x in parent2 if x not in common]
    return [child1,child2]

def single_point_crossover(parent1,parent2):
    i = random.randint(0,len(parent1)-1)
    child1 = parent1[:i+1] + parent2[i+1:]
    child2 = parent2[:i+1] + parent1[i+1:]
    return [child1,child2]

## 2. Mutation operators

def simple_swap(child): # Simple Swap (SS)
    i = random.randint(0,len(child)-1)
    j = random.randint(0,len(child)-1)
    child[i],child[j] = child[j],child[i]
    return child

def RSM(child): #Reverse Sequence Mutation: https://arxiv.org/ftp/arxiv/papers/1203/1203.3099.pdf#:~:text=In%20the%20reverse%20sequence%20mutation,covered%20in%20the%20previous%20operation.
    child = list(child) #returning a copy
    i = random.randint(0, len(child)-1)
    j = random.randint(0, len(child)-1)
    if j < i:
        i,j=j,i
    while i < j:
        child[i], child[j] = child[j], child[i]
        i += 1
        j -= 1
    return child

def bitwise_mutation(child,p_mut=0.5): # Bitwise mutation
    child = list(child)
    for i in range(len(child)):
        r = random.random()
        if r <= p_mut:
            child[i] = abs(child[i]-1)
    return child