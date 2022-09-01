"""Exact model: Generation of team combinations and build of the exact models."""
import itertools
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import time
import sys
if __name__=="__main__":
    sys.path.insert(0,"..")
from general_functions import belbin_score, mbti_score


def generate_combinations(subset_students,mini,maxi): 
    """
    Generates all team combinations given the team sizes and a set of students.

    Args:
      subset_students (list): Numbers representing the student IDs.
      mini (int): Minimum team size.
      maxi (int): Maximum team size.
    
    Returns:
      combination_list (list): each element is a set of students forming the team.
    """
    combination_list = []
    for i in range(mini,maxi+1): # Considering all team sizes in the range [mini,maxi]
        combination_list2 = list(itertools.combinations(subset_students,i))
        combination_list += combination_list2
    combination_list = list(map(set,combination_list)) # Each group is a set now
    return combination_list

def delete_no_mates(combination_list,nomate_list):
    """
    Filters those teams with incompatible students.
    Applied after `generate_combinations` will return the feasible teams
    of the problem.

    Args:
      combination_list (list): each element is a set of students forming the team.
      nomate_list (int): Each element is a pair of incompatible students, represented
        as a set. Example: [{1,3}, {6,87}] 
    
    Returns:
      feasible_teams (list): each element is a set of students forming a feasible team.
    """
    feasible_teams = []
    for team in combination_list:
        mates = True
        for nomates in nomate_list:
            if nomates.intersection(team)==nomates: 
                # The team must not be considered if its intersection with the incompatible
                # pair is the same as this pair (meaning that this pair is in fact in the team)
                mates = False
                break
        if mates:
            feasible_teams.append(team)
    return feasible_teams

def use_pywrap_model(teams,students,compulsory,scores_d,score_f,restr_num_groups,sname="SCIP"):
    """
    Begins by creating a model found in the pywrap module (SCIP, CBC, GLOP...).
    Continues by filling the model with the objective function and constraints that
    are defined in `V. Sanchez-Anguix, J. M. Alberola, E. D. Val, A. Palomares, & M.-D. Teruel, 
    «Assessing the use of intelligent tools to group tourism students into teams: 
    a comparison between team formation strategies»` for team formation.
    Ends up solving the problem and giving the results obtained.


    Args:
      teams (list): Teams considered as "feasible" in the problem.
      students (list): Numbers representing the student IDs.
      compulsory (list): Each element is a pair of compulsory students, represented
        as a set. Example: [{1,3}, {6,87}] 
      scores_d (dict): Contains the team roles or personality traits as keys and the
        values are lists of the scores obtained by each student. The ith score in these
        lists should match with the ith student in the `students` list.
        Example 1: {“CW”:[13,5,89,…,7],…,”CF”:[1,5,2,…,77]} 
        Example 2: {“E/I”:[3,-5,-9,…,6],…,”J/P”:[1,5,2,…,-17]}
      score_f (int): if it is 1, Belbin heuristic will be used. If it is 2, MBTI will
        be used. It is presupposed that this argument is used correctly, this is, if
        `scores_d` has Belbin scores, the user will then select `score_f=1`
      restr_num_groups (dict): relates to a new family of constraints where there is
        a limitation on the minimum/maximum number of groups to form of a certain size. 
        Keys are the team sizes, values are the minimum/maximum number of them.
        Example: {3:[0,4],4:[0,7],5:[0,6]}
      sname (str, optional): Name of the solver to be used.
    
    Returns:
      solve_time (float): Resolution time
      result_string (str): Type of result (optimal, feasible, abnormal...)
      best_teams (list): Given solution.
      best_scores (list): Each element is the score obtained by each team in the solution.
      value (float): Objective function's value for the given solution.
    """
    solver = pywraplp.Solver.CreateSolver(sname) # Solver initialization
    teams_variables = []
    for team in teams: # 1 binary variable for each possible team
        v = solver.BoolVar(f"Team {team} is selected for the project")
        teams_variables.append(v) # Adding the variables to a list
        
    # Restriction 1. Complete partition
    restr = solver.Constraint(len(students),len(students), f"Complete partition")
    for i in range(len(teams)):
        team = teams[i]
        team_size = len(team)
        var = teams_variables[i]
        restr.SetCoefficient(var,team_size) # Assigning team size to constraint
        
    # Restriction 2. Student j in only a team
    for stu in students: # For each student...
        restr = solver.Constraint(1,1, f"Student {stu} in only a team")
        for i in range(len(teams)): #...we look in which teams it is...
            if stu in teams[i]:
                var = teams_variables[i]
                restr.SetCoefficient(var,1) #...and we add to constraint in affirmative case
                
    # Restriction 3. Compulsory students in only a team
    set_st = set(students)
    for cjt in compulsory:
        if cjt.intersection(set_st)==cjt: # Verifies that considered students are in the set of all students
            restr = solver.Constraint(1,1, f"Compulsory students: {cjt}")
            for i in range(len(teams)):
                team = teams[i]
                if team.intersection(cjt): # If intersection between team and compulsory pair is not null, someone is in that team
                    var = teams_variables[i]
                    restr.SetCoefficient(var,1)

    # Restriction 4. Minimum and maximum number of groups of certain size
    for group_size, reslist in restr_num_groups.items():
        restr = solver.Constraint(reslist[0],reslist[1], f"Number of size {group_size} restriction")
        for i in range(len(teams)):
            team = teams[i]
            if len(team) == group_size:
                var = teams_variables[i]
                restr.SetCoefficient(var,1)

    # Objective function
    objective = solver.Objective()
    objective.SetMaximization()
    score_list = []
    for i in range(len(teams)): # Calling the scoring function for each possible team
        variable = teams_variables[i]
        team = teams[i]
        if score_f==1:
            score = belbin_score(scores_d,team,students)
        elif score_f==2:
            score = mbti_score(scores_d,team,students)
        objective.SetCoefficient(variable, score) 
        score_list.append(score) # Also adding the scores in a list that can be reused later

    t1 = time.time()
    result_type = solver.Solve()
    solve_time = time.time()-t1

    best_scores,best_teams = [],[]
    value = -1
    if result_type == solver.ABNORMAL:
        result_string = "abnormal"
    elif result_type == solver.FEASIBLE or result_type == solver.OPTIMAL:
        # If the solution is optimal or feasible, we loop through all teams and get those
        # that form the solution as well as their scores
        result_string = "feasible"
        value = objective.Value()
        for i in range(len(teams)):
            var = teams_variables[i]
            if var.solution_value()>0:
                team = teams[i]
                score = score_list[i]
                best_scores.append(score)
                best_teams.append(team)
        if result_type == solver.OPTIMAL:
            result_string = "optimal"        
    elif result_type == solver.INFEASIBLE:
        result_string = "infeasible"
    elif result_type == solver.NOT_SOLVED:
        result_string = "not solved"
    elif result_type == solver.UNBOUNDED:
        result_string = "unbounded"
    else:
        result_type = ""
    return solve_time,result_string,best_teams,best_scores,value

def use_SAT(teams,students,compulsory,scores_d,score_f,restr_num_groups,sname="SAT"):
    """
    Same behaviour as the `use_pywrap_model` function, but only uses the SAT solver.
    Please refer to that function to understand the inputs and outputs.
    """
    sat_model = cp_model.CpModel()
    teams_variables = []
    for team in teams: 
        v = sat_model.NewBoolVar(f"Team {team} is chosen")
        teams_variables.append(v) 

    # Restriction 1. Complete partition
    complete_partition = []
    for i in range(len(teams)):
        team = teams[i]
        team_size = len(team) 
        var = teams_variables[i]
        complete_partition.append(team_size*var)
    sat_model.Add(cp_model.LinearExpr.Sum(complete_partition)==len(students))

    # Restriction 2. Student j in only a team
    for student in students:
        eq1 = []
        for i in range(len(teams)):
            team = teams[i]
            if student in team:
                var = teams_variables[i]
                eq1.append(var)
        sat_model.Add(cp_model.LinearExpr.Sum(eq1)==1)

    #Restriction 3. Compulsory students in only a team
    set_st = set(students)
    for cjt in compulsory:
        if cjt.intersection(set_st)==cjt: 
            comp_res = []
            for i in range(len(teams)):
                team = teams[i]
                if team.intersection(cjt):
                    var = teams_variables[i]
                    comp_res.append(var)
            sat_model.Add(cp_model.LinearExpr.Sum(comp_res)==1)

    # Restriction 4. Minimum and maximum number of groups of certain size
    for group_size, reslist in restr_num_groups.items():
        restr = []
        for i in range(len(teams)):
            team = teams[i]
            if len(team) == group_size:
                var = teams_variables[i]
                restr.append(var)
        sat_model.Add(reslist[0] <= cp_model.LinearExpr.Sum(restr))
        sat_model.Add(cp_model.LinearExpr.Sum(restr) <= reslist[1])

    # Objective function
    obj = []
    score_list = []
    for i in range(len(teams)):
        team = teams[i]
        if score_f==1:
            score = int(belbin_score(scores_d,team,students)*8)
        elif score_f==2:
            score = int(mbti_score(scores_d,team,students)*8)
        score_list.append(score/8)
        var = teams_variables[i]
        obj.append(score*var)
    sat_model.Maximize(cp_model.LinearExpr.Sum(obj))

    solver = cp_model.CpSolver()
    # solver.parameters.num_search_workers=8
    t1 = time.time()
    result_type = solver.Solve(sat_model)
    solve_time = time.time()-t1

    best_scores,best_teams = [],[]
    value = -1
    if result_type == cp_model.UNKNOWN:
        result_string = "unknown"
    elif result_type == cp_model.MODEL_INVALID:
        result_string = "model_invalid"
    elif result_type == cp_model.FEASIBLE or result_type == cp_model.OPTIMAL:
        result_string = "feasible"
        if result_type == cp_model.OPTIMAL:
            result_string = "optimal"
        value = 0
        for i in range(len(teams)):
            var = teams_variables[i]
            if solver.Value(var)>0:
                team = teams[i]
                score = score_list[i]
                best_scores.append(score)
                best_teams.append(team)
                value += score           
    elif result_type == cp_model.INFEASIBLE:
        result_string = "infeasible"
    else:
        result_type = ""
    return solve_time,result_string,best_teams,best_scores,value