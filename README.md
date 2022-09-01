# Exact-models-and-metaheuristics-for-team-formation-TFG
This repository includes the code of all solutions implemented in my bachelor's thesis.
This main page has a script with some general functions for accessing the data and creating instances for experimentation and three folders.  

The [*data*](https://github.com/GonxaTroll/Exact-models-and-metaheuristics-for-team-formation-TFG_Code/tree/main/data) folder contains the anonymized data of some students, which is used for creating the instances.  

The [*exact_model_dir*](https://github.com/GonxaTroll/Exact-models-and-metaheuristics-for-team-formation-TFG_Code/tree/main/exact_model_dir) has the implementation of the exact methods, which are open-source, with ORTools. In case you want to make experiments, you can use *student_experimentation.py*. A detailed explanation on how to configure those experiments is given in that same folder.  

The [*genetic_algorithm*](https://github.com/GonxaTroll/Exact-models-and-metaheuristics-for-team-formation-TFG_Code/tree/main/genetic_algorithm) directory has the implementation of a Genetic Algorithm adapted to the constraints of the problem of team formation. You can make experiments with the *ga_experiments.py* script, check that folder for further instructions on how to make experiments or just execute the algorithm with your data.