# Exact models

In this folder you can find the implementation of the exact methods for solving the problem of team formation. You can see an example on how to use these at *exact_model_example_usage.ipynb*, which recreates a simple process of using a couple of solvers for computing the result of best team formation in a classroom with a reduced number of students. Concretely, the *use_pywrap_model* and *use_SAT* functions in the *exact_model.py* module are used to generate these results, and their inputs are described in the corresponding module.

You can also find the script that was used to experiment with the different solvers and served as a baseline to choose which one of them was the best for each type of problem. In case you want to make some experiments too, you can call the program by passing the following arguments (you can check the help when executing the program too):

student_experimentation.py scoring_function class_size [-solver SOLVER] [-repsize REPSIZE] [-expreps EXPREPS] [--force_same_sizes] [-group_size GROUP_SIZE] 
* scoring_function: "belbin" or "mbti" 
* class_size: Number of students (size of an instance)
* solver: solver to use. Possible values: "SCIP", "CBC", "BOP", "SAT". By default the program uses all except BOP if this argument is not included.
* repsize: Number of problem instances that are created and saved, 30 by default.
* expreps: Number of repetitions of the experiment.
* force_same_sizes: Wether or not to use only a team size. By default, teams can have betweeen 3 and 5 students, and currently there not exists and argument to control the minimum and maximum size.
* group_size: When forcing the team size, a specific team size. If not specified, all team sizes will be tried.
