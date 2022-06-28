import numpy as np
import re
import pandas as pd

#genera reps muestras de estudiantes de tamaño subset_size (tamaño de clases) y las guarda en fichero de texto si se pide como argumento
def generate_random_selection(student_list,subset_size,reps=30,savetxt = False,filename=""): #para los experimentos se generaban 30 clases distintas
    matrix = []
    for i in range(1,reps+1): #reps muestras
        choice = np.random.choice(student_list,size=subset_size,replace=False) #random choice de los estudiantes sin reemplazamiento
        matrix.append(choice)
    matrix = np.array(matrix) #matriz de muestras, cada fila es una muestra de subset_size estudiantes 
    if savetxt:
        pattern = ".+\.txt"
        if len(re.findall(pattern,filename)) == 0: #cadena dada no termina en .txt
            raise Exception("File name does not match the desired format")
        np.savetxt(fname = filename,X=matrix) #se guarda la matriz
    return matrix

#carga desde un fichero de texto una "matriz" de estudiantes (se basa en la función anterior)
def load_student_selection(filename):
    pattern = ".+\.txt"
    if len(re.findall(pattern,filename)) == 0:
        raise Exception("File name does not match the desired format")
    # return np.genfromtxt(filename,dtype=int)
    st = np.genfromtxt(filename)
    return st.astype(int)

#filtra aquellos equipos en los que hay estudiantes incompatibles
def delete_no_mates(combination_list,nomate_list):
    feasible_teams = []
    for team in combination_list:
        mates = True
        for nomates in nomate_list:
            if nomates.intersection(team)==nomates: #la intersección del equipo con el conjunto incompatible es este último conjunto, lo que quiere decir que no se debería considerar dicho grupo
                mates = False
                break
        if mates:
            feasible_teams.append(team)
    return feasible_teams

#Creación de función de Belbin
def belbin_score(belbin_scores_d,team,students):
    #implementer,coordinator,shaper,plant, resource-investigator, monitor-evaluator, teamworker,completer-finisher
    belbin_thresholds = {"CW":12,"CH":11,"SH":14,"PL":9,"RI":10,"ME":10,"TW":13,"CF":7} #lower limit of high scoring in each belbin role
    value = 0
    team = list(team)
    for role,role_list in belbin_scores_d.items(): #recorremos diccionario cuyas claves son los roles y los valores son listas con el score de cada estudiante
        for student in team:
            st_index = students.index(student) #encontramos al estudiante en la lista de todos los estudiantes
            if role_list[st_index] >= belbin_thresholds[role]: #sumamos 1 a f-obj si su score supera el threshold
                value += 1
                break
    return value/8

#Creación de función de Myers-Briggs
def mbti_score(mb_scores_d,team,students):
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
        if pos == 0 or neg == 0: #todos los miembros tienen una misma variante de personalidad (p.ej personalidad)
            value += 0
        elif pos == 1 or neg == 1: #hay 1 con un tipo distinto
            value += 1
        else:
            value += 2 #otro caso, se añade 2 a la función objetivo
    return value/8

#filtra los datos totales según la función de puntuación y genera muestras aleatorias si no existen ya
def subset_generate_data(data, score_f, subset_size, repsize, path=""):
    name = f"as_teamsize-{subset_size}-repsize-{repsize}.txt"
    small_data = data.loc[(data["has_belbin"]==1) & (data["has_mbti"]==1)] #filtrado a estudiantes que tengan ambos scores
    # small_data = small_data.sort_values(by="student_id") #ya están los datos ordenados, no hace falta
    full_small_students = list(small_data["student_id"])
    name = path + name
    try:
        rep_subset_students = load_student_selection(name)
    except:
        rep_subset_students = generate_random_selection(full_small_students,subset_size,reps=repsize,savetxt=True,filename=name)
    if score_f == 1: #belbin score
        small_data = small_data.iloc[:,[0]+list(range(2,10))] #obtenemos las columnas asociadas a los roles            
    elif score_f == 2: #proceso similar a lo anterior (mbti score en este caso)
        small_data = small_data.iloc[:,[0]+list(range(10,13+1))]
    return small_data, rep_subset_students #devuelve los datos filtrados y la matriz de muestras aleatorias

def get_scores_subset_students(small_data,subset_students):
    subset_small_data = small_data.loc[small_data["student_id"].isin(subset_students)] #se filtra los estudiantes completos según la muestra
    subset_small_data_large = pd.melt(subset_small_data,id_vars="student_id")
    scores_dict = {} #obtención de los scores en un diccionario para fácil acceso (clave=rol/personalidad,valores=listas con puntuaciones de cada estudiante)
    for key in pd.unique(subset_small_data_large["variable"]):
        key_data = subset_small_data_large[(subset_small_data_large["variable"]==key)]
        scores_dict[key] = list(key_data["value"])
    return list(subset_small_data["student_id"]), scores_dict #scores y estudiantes asociados a las listas de scores