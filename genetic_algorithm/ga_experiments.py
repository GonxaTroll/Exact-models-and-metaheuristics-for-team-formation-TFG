import pandas as pd
import argparse
import sys
from ast import literal_eval
sys.path.insert(0,"..")
from general_functions import *
import itertools
import time
from GA import *
import os

# Grid search of best operators/hyperparameters
# 70% of classes randomly generated for training
# 10 repetitions (random populations) per class
# Same population for each configuration and each scoring function


class EXPGA(StudentGA):
    def training_config(self,configuration,initial_population,experiment_file_path, optimal_solution):
        full_name = f"{experiment_file_path}_config-{configuration}"
        final_variables = ["full_name","experiment_name","configuration", "best_fitness",
        "iterations", "final_time", "final_population","optimal_solution"]
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

        population = initial_population
        generated_children = len(population)//2

        fitnesses = [self.get_fitness(x) for x in population]
        best_fitness_individual_iteration = max(fitnesses)
        best_individual_iteration = population[fitnesses.index(best_fitness_individual_iteration)]
        iteration_data.append([full_name,experiment_file_path,configuration, best_fitness_individual_iteration
            ,0,0,"pop",optimal_solution])
        if best_fitness_individual_iteration>=optimal_solution:
            result_data = [[full_name,experiment_file_path,str(configuration), best_fitness_individual_iteration,
                                0, 0, population,optimal_solution]]
            result_data = pd.DataFrame(result_data,columns=final_variables)
            iteration_data = pd.DataFrame(iteration_data,columns=final_variables)
            return result_data,iteration_data
    
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
            iteration_data.append([full_name,experiment_file_path,str(configuration), best_fitness_individual_iteration, 
                                   iterations,end_iteration_time, "pop",optimal_solution])

            if best_fitness_individual_iteration <= best_last_fitness:
                repeated_iterations += 1
            else:
                repeated_iterations = 0
            best_last_fitness = best_fitness_individual_iteration
            if best_fitness_individual_iteration >= optimal_solution or iterations==500 or repeated_iterations == 100: 
                final_time = time.time()-ini_time
                result_data = [[full_name,experiment_file_path,str(configuration), best_fitness_individual_iteration,
                                iterations, final_time, population,optimal_solution]]
                result_data = pd.DataFrame(result_data,columns=final_variables)
                iteration_data = pd.DataFrame(iteration_data,columns=final_variables)
                iteration_data.drop(columns=["final_population"],inplace=True)
                return result_data,iteration_data

#### MAIN PROGRAM
data = pd.read_csv("../data/student_data_anon.csv",delimiter=";")
nomate_list = [] 
nomate_list = list(map(set,nomate_list))
compulsory = []
compulsory = list(map(set,compulsory))
restr_num_groups = {}

# CLI arguments. Heuristic and class size are mandatory.
parser = argparse.ArgumentParser()
parser.add_argument("scoring_function",help="Use 'belbin' or 'mbti' heuristic")
parser.add_argument("class_size",help="Sample size of students")
parser.add_argument("-repsize", help = "Number of the generated samples",default=30,type=int)
parser.add_argument("-expreps", help = "Times for repeating the experiment",default=10,type=int)
parser.add_argument("-min_group_size",help="When forced teams, specify only a group size",type=int)
parser.add_argument("-max_group_size",help="When forced teams, specify only a group size",type=int)
parser.add_argument("--test",help="Use best configuration in test set",action="store_true")
args = parser.parse_args()

score_name = args.scoring_function
class_size = int(args.class_size)

if score_name == "belbin":
    score_f = 1
elif score_name == "mbti":
    score_f = 2
else:
    raise Exception("Scoring function not identified")

repsize = args.repsize
expreps = args.expreps
test = args.test

# (INCLUIR EN LA CLASE UNA FORMA DE QUE SI NO SE PUEDEN FORMAR DIVISIONES VÁLIDAS DEVUELVA ERROR)
data = pd.read_csv("../data/student_data_anon.csv",delimiter=";")
small_data, rep_subset_students  = subset_generate_data(data, score_f, class_size, repsize,path="../data/")

try: 
    with open(f"training_teamsize-{class_size}_repsize-{repsize}.txt","r") as training_file:
        training_instance_indices = literal_eval(training_file.read())
    with open(f"test_teamsize-{class_size}_repsize-{repsize}.txt","r") as test_file:
        test_instance_indices = literal_eval(test_file.read())
except:
    training_instance_indices = sorted(np.random.choice([x for x in range(repsize)],size=round(0.7*repsize),replace=False))
    test_instance_indices = [x for x in range(repsize) if x not in training_instance_indices]
    with open(f"training_teamsize-{class_size}_repsize-{repsize}.txt","w") as training_file:
        training_file.write(str(training_instance_indices))
    with open(f"test_teamsize-{class_size}_repsize-{repsize}.txt","w") as test_file:
        test_file.write(str(test_instance_indices))

ming,maxg = args.min_group_size, args.max_group_size 
if ming==maxg:
    strg = str(ming)
else:
    strg = f"{ming}-{maxg}"
exact_results = pd.read_csv("../exact_model_dir/Experiment results/experiment_data.csv")
exact_results = exact_results.loc[(exact_results["score_type"]==score_name) &
                                  (exact_results["class_size"]==class_size) &
                                  (exact_results["group_size"]==strg)]

if not test:
    partition_type = "training"
    try:
        with open("tuning_configurations.txt","r") as config_file:
            configurations = [literal_eval(line) for line in config_file]
    except: 
        with open("tuning_configurations.txt","w") as config_file:
            configurations = list(itertools.product(
            *[["NWOX","CX"],[0.7,0.9],
            ["InheritMask"],
            ["RSM","SS"],[0.2,0.4],
            ["DivisionSelect","RandomisedMasking"],
            ["RWS_sp-1.5"],
            ["EL"]]))
            for config in configurations:
                config_file.write(str(config)+"\n")
else:
    partition_type = "test"
    dic_configurations = {"belbin-20-3to5": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'RandomisedMasking', 'RWS-sp-1.5', 'EL', 100)],
                          "belbin-30-3to5": [('CX', 0.7, 'InheritMask', 'SS', 0.4, 'RandomisedMasking', 'RWS-sp-1.5', 'EL', 50)],
                          "belbin-40-3to5": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'RandomisedMasking', 'RWS-sp-1.5', 'EL', 100)],
                          "belbin-20-4to4": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 50)],
                          "belbin-20-5to5": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 50)],
                          "belbin-30-3to3": [('CX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 50)],
                          "belbin-30-5to5": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 50)],
                          "belbin-40-4to4": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 100)],
                          "belbin-40-5to5": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL',100)],
                          "belbin-60-3to3": [('CX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 50)],
                          "belbin-60-4to4": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 50)],
                          "mbti-20-3to5": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'RandomisedMasking', 'RWS-sp-1.5', 'EL',50)],
                          "mbti-30-3to5": [('NWOX', 0.9, 'InheritMask', 'SS', 0.2, 'DivisionSelect', 'RWS-sp-1.5', 'EL',100)],
                          "mbti-40-3to5": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'RandomisedMasking', 'RWS-sp-1.5', 'EL',50)],
                          "mbti-20-4to4": [('NWOX', 0.9, 'InheritMask', 'RSM', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL',50)],
                          "mbti-20-5to5": [('NWOX', 0.9, 'InheritMask', 'RSM', 0.2, 'DivisionSelect', 'RWS-sp-1.5', 'EL',50)],
                          "mbti-30-3to3": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL',50)],
                          "mbti-30-5to5": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL',50)],
                          "mbti-40-4to4": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 50)],
                          "mbti-40-5to5": [('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 50)],
                          "mbti-60-3to3": [('CX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 50)],
                          "mbti-60-4to4": [('CX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 50)],
                          } 

try:
    experiment_data = pd.read_csv(f"{partition_type}_data/{score_name}-{class_size}-{ming}to{maxg}-{partition_type}_experiments_final.csv")
    experiment_it_data = pd.read_csv(f"{partition_type}_data/{score_name}-{class_size}-{ming}to{maxg}-{partition_type}_experiments_it.csv")
except:
    cols = ["full_name","experiment_name","configuration", "best_fitness",
            "iterations", "final_time", "final_population","optimal_solution"]
    experiment_data = pd.DataFrame(columns=cols)
    experiment_it_data = pd.DataFrame(columns=cols).drop(columns=["final_population"])

if not test:
    for ins in training_instance_indices:
        optimal_sol = exact_results["score"].loc[exact_results["instance"]==ins+1].iloc[0]
        for population_size in [50,100]:
            list_full_data,list_iteration_data = [],[]
            for rep in range(1,11):
                subset_students = rep_subset_students[ins]
                subset_students,scores_dict = get_scores_subset_students(small_data,subset_students)
                genetic = EXPGA(subset_students,scores_dict,compulsory,nomate_list,ming,maxg,restr_num_groups,scoring_type=score_f)
                experiment_name = f"classize-{class_size}_instance-{ins+1}_repetition-{rep}"
                experiment_name = f"{experiment_name}_population-{population_size}_groupsize-{ming}to{maxg}"
                filename_pop = f"instance-{ins+1}_repetition-{rep}_population-{population_size}"
                try:
                    population = genetic.load_random_population(f"populations/classize-{class_size}/{ming}to{maxg}/{filename_pop}.txt")
                except:
                    if not os.path.exists(f"populations/classize-{class_size}/{ming}to{maxg}"):
                        os.makedirs(f"populations/classize-{class_size}/{ming}to{maxg}",exist_ok=False)
                    genetic.save_random_population(f"populations/classize-{class_size}/{ming}to{maxg}/{filename_pop}.txt")
                    population = genetic.load_random_population(f"populations/classize-{class_size}/{ming}to{maxg}/{filename_pop}.txt")
                experiment_name = f"{score_name}_{experiment_name}"
                for configuration in configurations:
                    full_name = f"{experiment_name}_config-{configuration}"
                    if full_name not in experiment_data["full_name"].values:
                        full_data,iteration_data = genetic.training_config(configuration,population,experiment_name,optimal_sol)
                        list_full_data.append(full_data)
                        list_iteration_data.append(iteration_data)

            if len(list_full_data) != 0:
                full_data = pd.concat(list_full_data)
                iteration_data = pd.concat(list_iteration_data)

                experiment_data = pd.concat([experiment_data,full_data])
                experiment_it_data = pd.concat([experiment_it_data,iteration_data])

                experiment_data.to_csv(f"{partition_type}_data/{score_name}-{class_size}-{ming}to{maxg}-{partition_type}_experiments_final.csv",index=False)
                experiment_it_data.to_csv(f"{partition_type}_data/{score_name}-{class_size}-{ming}to{maxg}-{partition_type}_experiments_it.csv",index=False)
else:
    best_configs = dic_configurations[f"{score_name}-{class_size}-{ming}to{maxg}"]
    for ins in test_instance_indices:
        optimal_sol = exact_results["score"].loc[exact_results["instance"]==ins+1].iloc[0]
        for config in best_configs:
            list_full_data,list_iteration_data = [],[]
            for rep in range(1,11):
                subset_students = rep_subset_students[ins]
                subset_students,scores_dict = get_scores_subset_students(small_data,subset_students)
                genetic = EXPGA(subset_students,scores_dict,compulsory,nomate_list,ming,maxg,restr_num_groups,scoring_type=score_f)
                
                population_size = config[8]

                experiment_name = f"classize-{class_size}_instance-{ins+1}_repetition-{rep}"
                experiment_name = f"{experiment_name}_population-{population_size}_groupsize-{ming}to{maxg}"
                filename_pop = f"instance-{ins+1}_repetition-{rep}_population-{population_size}"
                try:
                    population = genetic.load_random_population(f"populations/classize-{class_size}/{ming}to{maxg}/{filename_pop}.txt")
                except:
                    if not os.path.exists(f"populations/classize-{class_size}/{ming}to{maxg}"):
                        os.makedirs(f"populations/classize-{class_size}/{ming}to{maxg}",exist_ok=False)
                    genetic.save_random_population(f"populations/classize-{class_size}/{ming}to{maxg}/{filename_pop}.txt")
                    population = genetic.load_random_population(f"populations/classize-{class_size}/{ming}to{maxg}/{filename_pop}.txt")
                experiment_name = f"{score_name}_{experiment_name}"
                
                full_name = f"{experiment_name}_config-{config}"
                if full_name not in experiment_data["full_name"].values:
                    full_data,iteration_data = genetic.training_config(config,population,experiment_name,optimal_sol)
                    list_full_data.append(full_data)
                    list_iteration_data.append(iteration_data)

            if len(list_full_data) != 0:
                full_data = pd.concat(list_full_data)
                iteration_data = pd.concat(list_iteration_data)

                experiment_data = pd.concat([experiment_data,full_data])
                experiment_it_data = pd.concat([experiment_it_data,iteration_data])

                experiment_data.to_csv(f"{partition_type}_data/{score_name}-{class_size}-{ming}to{maxg}-{partition_type}_experiments_final.csv",index=False)
                experiment_it_data.to_csv(f"{partition_type}_data/{score_name}-{class_size}-{ming}to{maxg}-{partition_type}_experiments_it.csv",index=False)
