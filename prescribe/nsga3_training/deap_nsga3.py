from deap import tools, creator, base, algorithms
from math import factorial
import numpy
import _pickle as pickle
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import argparse
import sys
import os
import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simple RF model.''')

parser.add_argument('--adjusted_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')
parser.add_argument('--temp_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to data file with monthly temperatures.')
parser.add_argument('--start_date', nargs=1, type= str,
                  default=sys.stdin, help = 'Date to start from.')
parser.add_argument('--train_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to include in fitting.')
parser.add_argument('--forecast_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to forecast.')
parser.add_argument('--threshold', nargs=1, type= float,
                  default=sys.stdin, help = 'Threshold.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

###FUNCTIONS###
def setup_nsga3(NOBJ, NDIM, P, BOUND_LOW, BOUND_UP, NGEN, CXPB, MUTPB):
    #https://deap.readthedocs.io/en/master/examples/nsga3.html
    #Problem definition
    '''
    First we need to define the problem we want to work on. We will use the first problem tested in the paper,
    3 objectives DTLZ2 with k = 10 and p = 12. We will use pymop for problem implementation as it
    provides the exact Pareto front that we will use later for computing the performance of the algorithm.
    https://www.egr.msu.edu/~kdeb/papers/k2012009.pdf
    Test problem: https://tik-old.ee.ethz.ch/file//3ec604450bf683daaf27f9027e69f44d/DTLZ2004a.pdf
    '''
    H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1)) #Number of models in the pareto front


    '''
    Then we define the various parameters for the algorithm, including the population size set
    to the first multiple of 4 greater than H, the number of generations and variation probabilities.
    If twelve divisions (P = 12) are considered for each objective axis,
    91 reference points will be created, according to equation (10).
    (3+12-1) choose 12 = 14 choose 12 = 14!/[(14-12)!12!]=14*13/2=91
    '''
    MU = int(H + (4 - H % 4))


    '''
    Next, NSGA-III selection requires a reference point set. The reference point set serves to
    guide the evolution into creating a uniform Pareto front in the objective space.
    '''
    ref_points = tools.uniform_reference_points(NOBJ, P)

    '''
    As in any DEAP program, we need to populate the creator with the type of individual we require for our optimization.
    In this case we will use a basic list genotype and minimization fitness.
    '''
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
    creator.create("Individual", list, fitness=creator.FitnessMin)

    '''
    Moreover, we need to populate the evolutionary toolbox with initialization, variation and selection operators.
    Note how we provide the reference point set to the NSGA-III selection scheme.
    '''

    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    toolbox = base.Toolbox()
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #Register the evaluation, mating, mutation and selection processes
    #Load models for evaluation
    load_model()
    toolbox.register("evaluate", evaluate_npis)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

def load_model():
    '''Load the model
    '''
    low_models = []
    high_models = []
    #Fetch intercepts and coefficients
    modeldir='/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/3_weeks/non_log/june_on'
    for i in range(5):
        try:
            low_models.append(pickle.load(open(modeldir+'/low/model'+str(i), 'rb')))
            high_models.append(pickle.load(open(modeldir+'/high/model'+str(i), 'rb')))
        except:
            print('Missing fold',i)

    #Make global
    global low_models
    global high_models

def get_eval_data(data_path):
    '''Get the evaluation data
    '''

    adjusted_data = pd.read_csv(data_path,
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     dtype={"RegionName": str,
                            "RegionCode": str,
                            "Country_index":int,
                            "Region_index":int},
                     error_bad_lines=False)
    adjusted_data['RegionName'] = adjusted_data['RegionName'].replace('0', np.nan)
    adjusted_data = adjusted_data.fillna(0)

    selected_features = ['C1_School closing',
                    'C2_Workplace closing',
                    'C3_Cancel public events',
                    'C4_Restrictions on gatherings',
                    'C5_Close public transport',
                    'C6_Stay at home requirements',
                    'C7_Restrictions on internal movement',
                    'C8_International travel controls',
                    'H1_Public information campaigns',
                    'H2_Testing policy',
                    'H3_Contact tracing',
                    'H6_Facial Coverings', #These first 12 are the ones the prescriptor will assign
                    'Country_index',
                    'Region_index',
                    'CountryName',
                    'RegionName',
                    'smoothed_cases',
                    'cumulative_smoothed_cases',
                    'death_to_case_scale',
                    'case_death_delay',
                    'gross_net_income',
                    'population_density',
                    'monthly_temperature',
                    'pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr',
                    'Urban population (% of total population)',
                    'Population ages 65 and above (% of total population)',
                    'GDP per capita (current US$)', 'Obesity Rate (%)', 'Cancer Rate (%)',
                    'Share of Deaths from Smoking (%)', 'Pneumonia Death Rate (per 100K)',
                    'Share of Deaths from Air Pollution (%)',
                    'CO2 emissions (metric tons per capita)',
                    'Air transport (# carrier departures worldwide)',
                    'population']

    NB_LOOKBACK_DAYS=21
    # Make predictions for each country,region pair
    #Set threshold for model selection
    threshold=1.8

def evalate_npis(individual):
    '''Evaluate the prescriptor by predicting the outcome using the
    pretrained predictor.
    '''


def train(seed=None):
    '''
    The main part of the evolution is mostly similar to any other DEAP example.
    The algorithm used is close to the eaSimple() algorithm as crossover and mutation are applied to
    every individual (see variation probabilities above). However, the selection is made from the
    parent and offspring populations instead of completely replacing the parents with the offspring.
    '''

    random.seed(seed)

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)


    return pop, logbook


##########MAIN###########

#Params for NSGA3
#Parse args
args = parser.parse_args()
adjusted_data = pd.read_csv(args.adjusted_data[0],
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str,
                        "Country_index":int,
                        "Region_index":int},
                 error_bad_lines=False)
adjusted_data = adjusted_data.fillna(0)
#Get the monthly temperature data
monthly_temperature = pd.read_csv(args.temp_data)
start_date = args.start_date[0]
train_days = args.train_days[0]
forecast_days = args.forecast_days[0]
world_area = args.world_area[0]
threshold = args.threshold[0]
outdir = args.outdir[0]


NOBJ = 2
NDIM = 13 #Number of dimensions for net (prescriptor)
P = 12  #Number of divisions considered for each objective axis
BOUND_LOW, BOUND_UP = 0.0, 1.0
NGEN = 400 #Number of generations to run
CXPB = 1.0 #The probability of mating two individuals.
MUTPB = 1.0 #The probability of mutating an individual.
setup_nsga3(NOBJ, NDIM, P, BOUND_LOW, BOUND_UP, NGEN, CXPB, MUTPB)

pop, logbook = train()
front = numpy.array([ind.fitness.values for ind in pop])
pdb.set_trace()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(front[:,0], front[:,1],front[:,2], c="b")
plt.axis("tight")
plt.show()
pdb.set_trace()
