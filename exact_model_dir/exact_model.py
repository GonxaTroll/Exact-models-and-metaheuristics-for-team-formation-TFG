import itertools
from numpy import size
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import time
import sys
if __name__=="__main__":
    sys.path.insert(0,"..")
from general_functions import belbin_score, mbti_score


#define el modelo entero (variables, restricciones, función objetivo...)

def generate_combinations(subset_students,mini,maxi): #devuelve una lista de conjuntos con todas las combinaciones
    combination_list = []
    for i in range(mini,maxi+1): #diferencia con anterior: se prueba todas las combinaciones de 3 a 5 alumnos por grupo (mini,maxi)
        combination_list2 = list(itertools.combinations(subset_students,i))
        combination_list += combination_list2
    combination_list = list(map(set,combination_list)) #los grupos (tuplas) se transforman a conjuntos para uso de funciones de filtrado
    return combination_list

def use_pywrap_model(teams,students,compulsory,scores_d,score_f,restr_num_groups,sname="SCIP"):
    solver = pywraplp.Solver.CreateSolver(sname) #inicialización del solver
    teams_variables = []
    for team in teams: #1 variable binaria por equipo posible
        v = solver.BoolVar(f"Team {team} is selected for the project")
        teams_variables.append(v) #se añade a esta lista
        
    #Restriction 1. Complete partition
    restr = solver.Constraint(len(students),len(students), f"Complete partition")
    for i in range(len(teams)):
        team = teams[i]
        team_size = len(team) #tamaño del equipo
        var = teams_variables[i]
        restr.SetCoefficient(var,team_size) #asignación de tamaño de equipo a la restricción
        
    #Restriction 2. Student j in only a team
    for stu in students: #para cada estudiante...
        restr = solver.Constraint(1,1, f"Student {stu} in only a team")
        for i in range(len(teams)): #...se mira en qué equipos está...
            if stu in teams[i]:
                var = teams_variables[i]
                restr.SetCoefficient(var,1) #...y se añade a la restricción en caso afirmativo
                
    #Restriction 3. Compulsory students in only a team
    set_st = set(students)
    for cjt in compulsory:
        if cjt.intersection(set_st)==cjt: #verifica que los estudiantes indicados estén dentro del conjunto total de estudiantes
            restr = solver.Constraint(1,1, f"Compulsory students: {cjt}")
            for i in range(len(teams)):
                team = teams[i]
                if team.intersection(cjt): #si la intersección del equipo y los compulsory no es nula, es porque alguno está en ese equipo, por lo que se añade a la restricción
                    var = teams_variables[i]
                    restr.SetCoefficient(var,1)

    #Restriction 4. Minimum and maximum number of groups of certain size
    for group_size, reslist in restr_num_groups.items():
        restr = solver.Constraint(reslist[0],reslist[1], f"Number of size {group_size} restriction")
        for i in range(len(teams)):
            team = teams[i]
            if len(team) == group_size:
                var = teams_variables[i]
                restr.SetCoefficient(var,1)
    ### Objective function
    objective = solver.Objective()
    objective.SetMaximization()
    score_list = []
    for i in range(len(teams)): #llamamos a la función de puntuación (belbin, mbti...) para cada posible equipo
        variable = teams_variables[i]
        team = teams[i]
        if score_f==1:
            score = belbin_score(scores_d,team,students)
        elif score_f==2:
            score = mbti_score(scores_d,team,students)
        objective.SetCoefficient(variable, score) #coeficiente función objetivo
        score_list.append(score) #se añade tamb a una lista de valores precomputados para llamar luego

    t1 = time.time()
    result_type = solver.Solve()
    solve_time = time.time()-t1 #apuntamos tiempo de resolución

    best_scores,best_teams = [],[]
    value = -1
    if result_type == solver.ABNORMAL:
        result_string = "abnormal"
    elif result_type == solver.FEASIBLE or result_type == solver.OPTIMAL:
        result_string = "feasible"
        value = objective.Value() #en caso de factible u optima, se recorre hasta encontrar los equipos solución y sus puntuaciones
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
    sat_model = cp_model.CpModel()
    teams_variables = []
    for team in teams: #1 variable binaria por equipo posible
        # v = model.NewIntVar(0,1,f"Team {team}")
        v = sat_model.NewBoolVar(f"Team {team} is chosen")
        teams_variables.append(v) #se añade a esta lista

    #Restriction 1. Complete partition
    complete_partition = []
    for i in range(len(teams)):
        team = teams[i]
        team_size = len(team) #tamaño del equipo
        var = teams_variables[i]
        complete_partition.append(team_size*var)
    sat_model.Add(cp_model.LinearExpr.Sum(complete_partition)==len(students))
    # model.Add(sum(complete_partition)==len(students))

    #Restriction 2.
    for student in students:
        eq1 = []
        for i in range(len(teams)):
            team = teams[i]
            if student in team:
                var = teams_variables[i]
                eq1.append(var)
        # model.Add(sum(eq1)==1) 
        sat_model.Add(cp_model.LinearExpr.Sum(eq1)==1)

    #Restriction 3. Compulsory students in only a team
    set_st = set(students)
    for cjt in compulsory:
        if cjt.intersection(set_st)==cjt: #verifica que los estudiantes indicados estén dentro del conjunto total de estudiantes
            comp_res = []
            for i in range(len(teams)):
                team = teams[i]
                if team.intersection(cjt): #si la intersección del equipo y los compulsory no es nula, es porque alguno está en ese equipo, por lo que se añade a la restricción
                    var = teams_variables[i]
                    comp_res.append(var)
            sat_model.Add(cp_model.LinearExpr.Sum(comp_res)==1)

    for group_size, reslist in restr_num_groups.items():
        restr = []
        for i in range(len(teams)):
            team = teams[i]
            if len(team) == group_size:
                var = teams_variables[i]
                restr.append(var)
        sat_model.Add(reslist[0] <= cp_model.LinearExpr.Sum(restr))
        sat_model.Add(cp_model.LinearExpr.Sum(restr) <= reslist[1])

    obj = []
    score_list = []
    for i in range(len(teams)):
        team = teams[i]
        if score_f==1:
            # score = belbin_score(scores_d,team,students)
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