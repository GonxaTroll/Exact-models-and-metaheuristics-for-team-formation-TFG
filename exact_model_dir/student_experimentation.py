import pandas as pd
import argparse
import sys
sys.path.insert(0,"..")
from exact_model import *
from general_functions import *

#### MAIN PROGRAM
data = pd.read_csv("../data/student_data_anon.csv",delimiter=";")
mini,maxi = 3,5 # Minimum and maximum team sizes (fixed to 3 and 5 in experimentation)
nomate_list = [] # Rest of restrictions not considered in experimentation
nomate_list = list(map(set,nomate_list))
compulsory = []
compulsory = list(map(set,compulsory))
restr_num_groups = {}

# CLI arguments. Heuristic and class size are mandatory.
parser = argparse.ArgumentParser()
parser.add_argument("scoring_function",help="Use 'belbin' or 'mbti' heuristic")
parser.add_argument("class_size",help="Sample size of students")
parser.add_argument("-solver",help="Used individual solver (default will be SCIP, CBC and SAT")
parser.add_argument("-repsize", help = "Number of the generated samples",default=30,type=int)
parser.add_argument("-expreps", help = "Times for repeating the experiment",default=5,type=int)
parser.add_argument("--force_same_sizes",help="If it is desired to force groups into same sizes (all possibilities will be tried)",action="store_true")
parser.add_argument("-group_size",help="When forced teams, specify only a group size",type=int)
args = parser.parse_args()

score_name = args.scoring_function
subset_size = int(args.class_size)
solver_names = args.solver

# for i in range(mini,maxi+1):
    # restr_num_groups[i] = [0,subset_size//i]

if score_name == "belbin":
    score_f = 1
elif score_name == "mbti":
    score_f = 2
else:
    raise Exception("Scoring function not identified")

repsize = args.repsize
expreps = args.expreps

if solver_names is None: # If the optional argument is not specified, all solvers are tried
    solver_names = ["SCIP","CBC","SAT"]
else:
    solver_names = [solver_names]

force_same_size_groups = args.force_same_sizes
if force_same_size_groups:
    possible = [x for x in range(mini,maxi+1) if subset_size % x == 0] # Only getting feasible team sizes
    group_size = args.group_size 
    if group_size is not None:
        if group_size in possible:
            possible = [group_size]
        else:
            raise Exception("Invalid group size for forming equal groups")

# Creating data to save the results or loading the experiment data in case they already exist
cols = ["class_size","instance","rep","solver","resolution_time","result_type","formed_teams","team_scores","score","score_type","name","group_size"]
try:
    experiment_data = pd.read_csv("Experiment results/experiment_data.csv")
except:
    experiment_data = pd.DataFrame(columns = cols)

# Subsetting the whole dataset for the sampled students
small_data, rep_subset_students = subset_generate_data(data, score_f, subset_size, repsize, path="../data/")
      
for solver_name in solver_names:
    if solver_name == "SAT":
        use_model = use_SAT
    else:
        use_model = use_pywrap_model
    instance = 0
    for subset_students in rep_subset_students: # For each generated sample of size `subset_size`
        print(subset_students)
        instance += 1
        subset_students,scores_dict = get_scores_subset_students(small_data,subset_students)
        for repetition in range(1,expreps+1): # 5 repetitions are performed in case results are not fully deterministic (solver issues...)
            experiment_name = f"{score_name}-Student_num-{subset_size}-instance-{instance}-rep{repetition}-{solver_name}"
            if force_same_size_groups: # If we want to have only a same team size... (ex, class of 30, 10 groups of 3)
                for i in possible: # For each possible team size i...
                    experiment_name2 = experiment_name + f"-groupsize-{i}"
                    print(experiment_name2)
                    if experiment_name2 not in list(experiment_data["name"]): # Evaluating that the experiment has not been already performed
                        combination_list = generate_combinations(subset_students,i,i) # All student combinations taken from i to i
                        combination_list = delete_no_mates(combination_list,nomate_list)
                        resolution_time, result_string, best_teams, best_scores,value = use_model(combination_list, subset_students, compulsory, scores_dict, score_f, restr_num_groups,sname=solver_name)
                        new_values = [[subset_size,instance,repetition,solver_name,resolution_time,result_string,best_teams,best_scores,value,score_name,experiment_name2,i]]
                        new_data = pd.DataFrame(new_values,columns=cols)
                        experiment_data = pd.concat([experiment_data,new_data])
                        experiment_data.to_csv("Experiment results/experiment_data.csv",index=False) # Saving intermediate results
            else: # Same proceeding as the previous case
                print(experiment_name)
                if experiment_name not in list(experiment_data["name"]):
                    combination_list = generate_combinations(subset_students,mini,maxi)
                    combination_list = delete_no_mates(combination_list,nomate_list)
                    resolution_time, result_string, best_teams, best_scores,value = use_model(combination_list, subset_students, compulsory, scores_dict, score_f, restr_num_groups,sname=solver_name)
                    new_values = [[subset_size,instance,repetition,solver_name,resolution_time,result_string,best_teams,best_scores,value,score_name,experiment_name,f"{mini}-{maxi}"]]
                    new_data = pd.DataFrame(new_values,columns=cols)
                    experiment_data = pd.concat([experiment_data,new_data])
                    experiment_data.to_csv("Experiment results/experiment_data.csv",index=False)

# Call examples to the program:

# python student_experimentation.py belbin 50 (Experiments with all solvers for Belbin in 30 classes of 50 students)
# python student_experimentation.py belbin 15 -solver BOP --force_same_sizes (Experiments with BOP for Belbin in 30 classes of 15 students each forcing same group sizes)