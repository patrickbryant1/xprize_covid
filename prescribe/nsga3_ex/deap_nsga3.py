from deap import tools

#https://deap.readthedocs.io/en/master/examples/nsga3.html
#Problem definition
'''
First we need to define the problem we want to work on. We will use the first problem tested in the paper,
3 objectives DTLZ2 with k = 10 and p = 12. We will use pymop for problem implementation as it
provides the exact Pareto front that we will use later for computing the performance of the algorithm.
'''
PROBLEM = "dtlz2"
NOBJ = 3
K = 10
NDIM = NOBJ + K - 1
P = 12
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
BOUND_LOW, BOUND_UP = 0.0, 1.0
problem = pymop.factory.get_problem(PROBLEM, n_var=NDIM, n_obj=NOBJ)
