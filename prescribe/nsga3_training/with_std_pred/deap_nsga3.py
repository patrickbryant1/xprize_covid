#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from deap import tools, creator, base, algorithms
from math import factorial
import numpy
import _pickle as pickle
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import os
from xprize_predictor import XPrizePredictor
import pdb

#Inser predictor path. NOTE! This will have to be absoulate in the sandbox
sys.path.insert(0, "../../standard_predictor/")
import predict
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simple RF model.''')

parser.add_argument('--pred_dir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to prediction directory.')
parser.add_argument('--ip_costs', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to data file with ip costs per region.')
parser.add_argument('--start_date', nargs=1, type= str,
                  default=sys.stdin, help = 'Date to start from.')
parser.add_argument('--forecast_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to forecast.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

###FUNCTIONS###
def load_model(start_date,lookback_days):
    '''Load the standard predictor
    '''
    NPI_COLUMNS = ['GeoID',
                   'Date',
                   'C1_School closing',
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
                   'H6_Facial Coverings']

    # Fixed weights for the standard predictor.
    MODEL_WEIGHTS_FILE = './trained_model_weights.h5'
    DATA_FILE = './OxCGRT_latest.csv'
    data = pd.read_csv(DATA_FILE,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                   "RegionCode": str},
                            error_bad_lines=False)
    # GeoID is CountryName / RegionName
    # np.where usage: if A then B else C
    data["GeoID"] = np.where(data["RegionName"].isnull(),
                                  data["CountryName"],
                                  data["CountryName"] + ' / ' + data["RegionName"])

    inp_data = data[(data.Date >= start_date) & (data.Date <= (start_date + np.timedelta64(lookback_days, 'D')))]
    npis_inp_data = inp_data[NPI_COLUMNS]
    case_inp_data =  data['ConfirmedCases']
    pdb.set_trace()

    predictor = XPrizePredictor(MODEL_WEIGHTS_FILE, DATA_FILE)


def setup_nsga3(NOBJ, NDIM, P, BOUND_LOW, BOUND_UP, CXPB, MUTPB, start_date, lookback_days):
    #https://deap.readthedocs.io/en/master/examples/nsga3.html
    #Problem definition
    '''
    #Number of models in the pareto front
    '''
    H = 100 #factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))


    '''
    Then we define the various parameters for the algorithm, including the population size set
    to the first multiple of 4 greater than H, the number of generations and variation probabilities.
    If twelve divisions (P = 12) are considered for each objective axis and there are 3 objectives,
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
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM) #12 interventions
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #Register the evaluation, mating, mutation and selection processes
    #Load models for evaluation
    load_model(start_date, lookback_days)
    toolbox.register("evaluate", evaluate_npis)
    #eta = Crowding degree of the crossover.
    #A high eta will produce children resembling to their parents, while a small eta will produce solutions much more different.
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
    #indpb – Independent probability for each attribute to be mutated.
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    return toolbox, creator, MU



def evaluate_npis(individual):
    '''Evaluate the prescriptor by predicting the outcome using the
    pretrained predictor.
    '''
    #Get copy
    
    X_ind = np.copy(npis_inp_data)
    #Convert to array and reshape
    individual = np.array(individual)
    individual = np.reshape(individual,(12,NUM_WEIGHTS,NUM_LAYERS))
    #Prescribe and predict for n 21 day periods
    obj1 = 0 #Cumulative preds
    obj2 = 0 #Cumulative issued NPIs
    #Start and end dates
    current_date=start_date+ np.timedelta64(lookback_days, 'D')

    for n in range(2): #2 21 day periods, which should be sufficient to observe substantial changes
        #Get prescriptions and scale with weights
        prev_ip = X_ind[:,:12]*ip_weights
        #Get cases in last period
        prev_cases = X_ind[:,12]
        #Multiply prev ip with the 2 prescr weight layers of the individual

        prescr = prev_ip*individual[:,0,0]*individual[:,0,1]
        #Add the case focus
        prescr += np.array([prev_cases]).T*individual[:,1,0]*individual[:,1,1]
        #Now the prescr can't really increase based only on the prescr
        #The cases will have to be high for the ips to increase
        #Perhaps this is not such a bad thing
        #Make sure the prescr don't exceeed the npi maxvals
        prescr = np.minimum(prescr,ip_maxvals)
        X_ind[:,:12]=prescr

        #Generate the predictions
        pdb.set_trace()
        preds_df = predictor.predict(start_date, end_date, path_to_ips_file)

        #Add cases and NPI sums
        #Check where 0
        #zind = np.argwhere(X_ind[:,12]==0)
        #X_ind[:,12][zind]=all_preds[zind]
        #diff = all_preds/X_ind[:,12]
        obj1 += np.sum(all_preds)
        obj2 += np.sum(prescr)

        #Update X_ind
        X_ind[:,12]=all_preds

    return obj1, obj2



def train(seed,toolbox, creator,NGEN, CXPB, MUTPB):
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

ip_costs = pd.read_csv(args.ip_costs[0])
ip_costs['GeoID'] = ip_costs['CountryName'] + '__' + ip_costs['RegionName'].astype(str)
start_date = args.start_date[0]
lookback_days = args.lookback_days[0]
forecast_days = args.forecast_days[0]
outdir = args.outdir[0]

NOBJ = 2
NUM_WEIGHTS=2
NUM_LAYERS=2
NDIM = 12*NUM_WEIGHTS*NUM_LAYERS #Number of dimensions for net (prescriptor): 12 ips, 2 weights (1 for the ip, 1 for the cases), 2 layeres
P = 12  #Number of divisions considered for each objective axis
#If P divisions are considered for each objective axis, and
#there are N objective axes, (N+P-1) choose P reference points Will
#be created = (N+P-1)!/((N+P-1-P)!P!)
#If N=2 and P= 12 --> 13!/(12!*1!)=13

#Weight boundaries
BOUND_LOW, BOUND_UP = 0.0, 1.0
NGEN = 200 #Number of generations to run
CXPB = 1.0 #The probability of mating two individuals.
MUTPB = 1.0 #The probability of mutating an individual.

toolbox, creator, MU = setup_nsga3(NOBJ, NDIM, P, BOUND_LOW, BOUND_UP, CXPB, MUTPB,start_date,lookback_days)

pop, logbook = train(42,toolbox, creator,NGEN, CXPB, MUTPB)

#Get the pareto front
front = numpy.array([ind.fitness.values for ind in pop])
#Save
np.save(outdir+'population.npy',np.array(pop))
np.save(outdir+'front.npy',front)
#Plot
plt.scatter(front[:,0], front[:,1], c="b")
plt.xlabel('Cases')
plt.ylabel('Stringency')
plt.title('Pareto front')
plt.axis("tight")
plt.savefig(outdir+'pareto_front.png',format='png')
pdb.set_trace()
