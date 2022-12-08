# Modified WALKSAT Algorithm for solving satisfiability(SAT) problems

-This code implements WALKSAT algorithm for solving SAT problems.
-Instead of just using random assignments, it uses uniform probabilty to flip least false-clause-creating "v" variables in the model.
-I have uploaded both jupyter notebook and my python script separately.
-The code was run on 162 different inputs taken from data generated by me :) .
-The hyperparameters maxv, maxit, pflip can be easily changed by the user.Default values are p = .5 , maxv = 4 and maxit = 500 .(These values can be changed by changing values in the parameters of the function WALKsat )
-TO RUN CODE EXECUTE "sh run.sh".

-

# Requirements

pip install numpy
pip install pandas
pip install matplotlib

# Dataset

-The dataset is in the 'findata' folder.
-It contains 162 files containing SAT problems in DIMAC format.
-Clauses ends with 0.
-starts with a comment "c data created by urvil jivani."

# Results

- high disturbance in range 4-5 in the graph for(maxv=4,maxit=500,p=.5).
- high disturbance in range 4-5 in the graph for(maxv=3,maxit=500,p=.5).
- if we choose maxv = 5 computation time is more and results donot show much change .
- final graphs are in "mnratio_vs_log(avgIter).png" .
- this file gives us 2 graphs : on the left is mnratio_vs_avgIter and on the right is mnratio_vs_log(avgIter) .
- "important_info.csv" is created for information of each iteration .
- no_of_iter(gives no of times each file is iterated) and its results are stored in the file "important_info.csv".
- no_of_iter helps us to get smoother curves .
- if no_of_iter are around 5 then it take 15 - 20 min to execute on (maxv=4,maxit=500,p=.5)
- "important*info.csv" contains 2 columns for fileNumber and fileName(in the format "m=*,n=\_.cnf") and other columns are for each iteration of that file . Each iteration has 3 columns "iter*:time","iter*:iter","iter:m/n_ratio" .