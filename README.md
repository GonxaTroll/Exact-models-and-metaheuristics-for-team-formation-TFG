# Exact-models-and-metaheuristics-for-team-formation-TFG
This repository includes the code of all solutions implemented in my final degree dissertation
It consists of a script with some general functions for accessing the data and creating instances for experimentation and three folders.
The *data* folder contains the anonymized data of some students, which is used for creating the instances.
The *exact_model_dir* has the implementation of the exact methods, which are open-source, with ORTools.In case you want to make experiments, you can check *student_experimentation.py* and its parameters.
The *genetic_algorithm* directory has the implementation of a Genetic Algorithm adapted to the constraints of the problem of team formation. You can make experiments with the *ga_experiments.py* script, an example using the best configuration in the test set of instances would be:
python3 ga_experiments.py mbti 40 -min_group_size 3 -max_group_size 5 --test
All of the parameters can be found within that script.
