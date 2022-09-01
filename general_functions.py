import numpy as np
import re
import pandas as pd


def generate_random_selection(student_list,subset_size,reps=30,savetxt = False,filename=""):
    """
    Generates `reps` samples of students of `subset_size` size and saves them in a text
    file if desired.
    
    Args:
      student_list (list): Numbers representing the student IDs.
      subset_size: Number of students in each sample.
      reps (int): Number of samples generated, 30 by default (experimentation)
      savetxt (bool): Saving the sample of matrix into txt file or not.
      filename (str): Name of file where results should be saved.

    Returns:
      matrix (ndarray)
    """
    matrix = []
    for i in range(1,reps+1):
        choice = np.random.choice(student_list,size=subset_size,replace=False) # Random choice of students without replacement
        matrix.append(choice)
    matrix = np.array(matrix) # Sample matrix, each row is a sample of `subset_size` students 
    if savetxt:
        pattern = ".+\.txt"
        if len(re.findall(pattern,filename)) == 0: # String does not end in .txt
            raise Exception("File name does not match the desired format")
        np.savetxt(fname = filename,X=matrix) # Saving the matrix
    return matrix

def load_student_selection(filename):
    """Loads a text file containing a student matrix (based in previous function)"""
    pattern = ".+\.txt"
    if len(re.findall(pattern,filename)) == 0:
        raise Exception("File name does not match the desired format")
    st = np.genfromtxt(filename)
    return st.astype(int)

def belbin_score(belbin_scores_d,team,students):
    """
    Computes the Belbin score of a given team following the heuristic from
    `V. Sanchez-Anguix, J. M. Alberola, E. D. Val, A. Palomares, & M.-D. Teruel, 
    «Assessing the use of intelligent tools to group tourism students into teams: 
    a comparison between team formation strategies»`

    Args: 
      belbin_scores_d (dict): Team roles as keys and values are the scores of each student.
      team (set): 
      students (list): Numbers representing the student IDs.

    Returns:
      float
    """
    # Roles: Implementer, Coordinator, Shaper, Plant, Resource-Investigator, Monitor-Evaluator,
    # Teamworker, Completer-Finisher. Below are the lower limits of high scoring in each one.
    belbin_thresholds = {"CW":12,"CH":11,"SH":14,"PL":9,"RI":10,"ME":10,"TW":13,"CF":7} 
    value = 0
    team = list(team)
    for role,role_list in belbin_scores_d.items(): 
        for student in team:
            st_index = students.index(student) # Finding the student in the student list
            if role_list[st_index] >= belbin_thresholds[role]: # +1 to obj-func if score>=threshold
                value += 1
                break
    return value/8

def mbti_score(mb_scores_d,team,students):
    """Similar to `belbin_score` but with MBTI heuristic"""
    value = 0
    team = list(team)
    for pers,pers_list in mb_scores_d.items():
        neg,pos = 0,0
        for student in team:
            st_index = students.index(student)
            if pers_list[st_index]>0:
                pos += 1
            elif pers_list[st_index]<0:
                neg += 1
        if pos == 0 or neg == 0: # All members with same personality trait
            value += 0
        elif pos == 1 or neg == 1: # 1 with different type
            value += 1
        else:
            value += 2
    return value/8

def subset_generate_data(data, score_f, subset_size, repsize, path=""):
    """
    Filters complete data considering the scoring function and generates random samples
    if they do not exist yet.
    """
    name = f"as_teamsize-{subset_size}-repsize-{repsize}.txt"
    small_data = data.loc[(data["has_belbin"]==1) & (data["has_mbti"]==1)] # Only students with both scores
    # small_data = small_data.sort_values(by="student_id") # Data already sorted
    full_small_students = list(small_data["student_id"])
    name = path + name
    try:
        rep_subset_students = load_student_selection(name)
    except:
        rep_subset_students = generate_random_selection(full_small_students,subset_size,reps=repsize,savetxt=True,filename=name)
    if score_f == 1: # Belbin score
        small_data = small_data.iloc[:,[0]+list(range(2,10))] # Columns associated with those roles          
    elif score_f == 2: # MBTI score
        small_data = small_data.iloc[:,[0]+list(range(10,13+1))]
    return small_data, rep_subset_students # Filtered data and random sample matrix

def get_scores_subset_students(small_data,subset_students):
    """Subsets the whole dataset and also returns a dict of scores of students"""
    subset_small_data = small_data.loc[small_data["student_id"].isin(subset_students)] # Filtering complete data considering the sample of students
    subset_small_data_large = pd.melt(subset_small_data,id_vars="student_id")
    scores_dict = {} # Scores in a dict where keys=roles/traits, values=lists_of_scores
    for key in pd.unique(subset_small_data_large["variable"]):
        key_data = subset_small_data_large[(subset_small_data_large["variable"]==key)]
        scores_dict[key] = list(key_data["value"])
    return list(subset_small_data["student_id"]), scores_dict

def scores_table2dict(df):
    """Transforms a matrix of scores into a dictionary"""
    df = pd.melt(df,id_vars="student_id")
    scores_dict = {}
    for key in pd.unique(df["variable"]):
        key_data = df[(df["variable"]==key)]
        scores_dict[key] = list(key_data["value"])
    return scores_dict