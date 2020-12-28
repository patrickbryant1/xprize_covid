from deap import tools
import pymop

#https://deap.readthedocs.io/en/master/examples/nsga3.html
#Problem definition
'''
First we need to define the problem we want to work on. We will use the first problem tested in the paper,
3 objectives DTLZ2 with k = 10 and p = 12. We will use pymop for problem implementation as it
provides the exact Pareto front that we will use later for computing the performance of the algorithm.
https://www.egr.msu.edu/~kdeb/papers/k2012009.pdf
Test problem: https://tik-old.ee.ethz.ch/file//3ec604450bf683daaf27f9027e69f44d/DTLZ2004a.pdf
'''
PROBLEM = "dtlz2"
NOBJ = 3
K = 10
NDIM = NOBJ + K - 1
P = 12
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
BOUND_LOW, BOUND_UP = 0.0, 1.0
problem = pymop.factory.get_problem(PROBLEM, n_var=NDIM, n_obj=NOBJ)

'''
Then we define the various parameters for the algorithm, including the population size set
to the first multiple of 4 greater than H, the number of generations and variation probabilities.
'''
MU = int(H + (4 - H % 4))
NGEN = 400
CXPB = 1.0
MUTPB = 1.0

'''
Next, NSGA-III selection requires a reference point set. The reference point set serves to
guide the evolution into creating a uniform Pareto front in the objective space.

 If twelve divisions (d = 12) are considered for each objective axis,
 91 reference points will be created, according to equation (10).
 (3+12-1) choose 12 = 14 choose 12 = 14!/[(14-12)!12!]=14*13/2=91
'''
ref_points = tools.uniform_reference_points(NOBJ, P)

'''
As in any DEAP program, we need to populate the creator with the type of individual we require for our optimization.
In this case we will use a basic list genotype and minimization fitness.
'''
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin)
