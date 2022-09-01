# Genetic Algorithm

Folder containing the class with the genetic algorithm and experimentation with it. The arguments that it takes to construct an object of this class are very similar to those needed when computing the results of the exact models, you can take a look at the constructor of this class in *GA.py*. There is also a notebook that shows how to create an instance of this class and retrieve the results from the execution of this algorithm (via the *execute* method).

The experimentation code is also provided, check *ga_experiments.py* for that. An execution of this script could take the following arguments:  

ga_experiments.py scoring_function class_size [-repsize REPSIZE] [-expreps EXPREPS] [-min_group_size MIN_GROUP_SIZE] [-max_group_size MAX_GROUP_SIZE] [--test]   
* scoring_function: "belbin" or "mbti" 
* class_size: Number of students (size of an instance)
* repsize: Number of problem instances that are created and saved, 30 by default.
* expreps: Number of repetitions of the experiment.
* min_group_size: Minimum team size.
* max_group_size: Maximum team size.
* test: If you want to use the best configuration with the test set or not.
