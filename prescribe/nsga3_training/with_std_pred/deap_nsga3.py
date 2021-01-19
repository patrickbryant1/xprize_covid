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
import time
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
parser.add_argument('--lookback_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to lookback for forecast.')
parser.add_argument('--forecast_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to forecast.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

###FUNCTIONS###
def load_model(start_date,lookback_days,ip_costs):
    '''Load the standard predictor
    '''
    #Define global to use for all inds
    global X_prescr_inp, npis_data, ip_maxvals, ip_weights, predictor

    DATA_COLUMNS = ['CountryName',
                    'RegionName',
                    'GeoID',
                    'Date',
                    'smoothed_cases',
                    'ConfirmedCases',
                    'ConfirmedDeaths',
                    'population',
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

    IP_MAX_VALUES = {'C1_School closing': 3,
                    'C2_Workplace closing': 3,
                    'C3_Cancel public events': 2,
                    'C4_Restrictions on gatherings': 4,
                    'C5_Close public transport': 2,
                    'C6_Stay at home requirements': 3,
                    'C7_Restrictions on internal movement': 2,
                    'C8_International travel controls': 4,
                    'H1_Public information campaigns': 2,
                    'H2_Testing policy': 3,
                    'H3_Contact tracing': 2,
                    'H6_Facial Coverings': 4
                    }
    #Set ip maxvals
    ip_maxvals = np.array([*IP_MAX_VALUES.values()])

    # Fixed weights for the standard predictor.
    MODEL_WEIGHTS_FILE = './trained_model_weights.h5'
    DATA_FILE = '../../../data/adjusted_data.csv' #Preprocessed data
    data = pd.read_csv(DATA_FILE,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                   "RegionCode": str},
                            error_bad_lines=False)
    data["RegionName"]= data["RegionName"].replace('0',np.nan)
    data["GeoID"] = np.where(data["RegionName"].isnull(),
                                  data["CountryName"],
                                  data["CountryName"] + ' / ' + data["RegionName"])
    # GeoID is CountryName__RegionName (This they changed to "/", but I changed it back)
    # np.where usage: if A then B else C
    data = data[DATA_COLUMNS]
    #Normalize cases for prescriptor
    data['smoothed_cases']=data['smoothed_cases']/(data['population']/100000)
    #Get inp data for prescriptor
    inp_data = data[(data.Date >= start_date) & (data.Date <= (pd.to_datetime(start_date, format='%Y-%m-%d') + np.timedelta64(lookback_days-1, 'D')))]
    #Get only npi data
    npis_data = inp_data.drop(columns={'smoothed_cases','ConfirmedCases','ConfirmedDeaths'})
    prescr_inp_data =  inp_data.drop(columns={'ConfirmedCases','ConfirmedDeaths','population'})
    #Format prescr inp data for prescriptor
    #Get ip costs
    ip_weights = []
    X_prescr_inp = []
    for geo in prescr_inp_data.GeoID.unique():
        geo_data = prescr_inp_data[prescr_inp_data['GeoID']==geo]
        X_geo= np.average(geo_data[DATA_COLUMNS[-12:]],axis=0)
        X_geo = np.append(X_geo,np.median(geo_data['smoothed_cases']))
        X_prescr_inp.append(X_geo)
        #Get ip weights
        ip_weights.append(ip_costs[ip_costs['GeoID']==geo][DATA_COLUMNS[-12:]].values[0])
    #Convert to array
    X_prescr_inp = np.array(X_prescr_inp)
    ip_weights = np.array(ip_weights)
    #Load predictor
    predictor = XPrizePredictor(MODEL_WEIGHTS_FILE, './OxCGRT_latest.csv')


def setup_nsga3(NOBJ, NDIM, P, BOUND_LOW, BOUND_UP, CXPB, MUTPB, start_date, lookback_days,ip_costs):
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
    load_model(start_date, lookback_days,ip_costs)
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
    X_ind = np.copy(X_prescr_inp)
    npis_data_ind = npis_data.copy()
    #Convert to array and reshape
    individual = np.array(individual)
    individual = np.reshape(individual,(12,NUM_WEIGHTS,NUM_LAYERS))
    #Prescribe and predict for n 21 day periods
    obj1 = 0 #Cumulative preds
    obj2 = 0 #Cumulative issued NPIs
    #Start and end dates
    current_date=pd.to_datetime(start_date, format='%Y-%m-%d')+ np.timedelta64(lookback_days, 'D')

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
        #Distribute the prescriptions in the forecast period
        #Define the new dates
        new_dates = np.arange(current_date,current_date + np.timedelta64(forecast_days, 'D'), dtype='datetime64[D]')
        geo_i = 0
        for geo in npis_data_ind.GeoID.unique():
            geo_inds = npis_data_ind[npis_data_ind['GeoID']==geo].index
            #Assign new dates
            npis_data_ind.at[geo_inds,['Date']]=new_dates
            #Assign new prescriptions
            npis_data_ind.at[geo_inds,npis_data_ind.columns[-12:]]=prescr[geo_i,:]
            #I choose to keep these stable over a three week period as changing them
            #on e.g. a daily or weekly basis in various degrees will not only make them
            #hard to follow but also confuse the public
        #Generate the predictions
        #time
        tic = time.clock()
        preds_df = predictor.predict(current_date, current_date + np.timedelta64(forecast_days-1, 'D'),npis_data_ind)
        toc = time.clock()
        print(np.round((toc-tic)/60,2))
        #Add GeoID
        preds_df["GeoID"] = np.where(preds_df["RegionName"].isnull(),
                                      preds_df["CountryName"],
                                      preds_df["CountryName"] + ' / ' + preds_df["RegionName"])
        #Add the predictions to the next step
        median_case_preds = []
        for geo in npis_data_ind.GeoID.unique():
            geo_pred_ind = preds_df[preds_df['GeoID']==geo]
            geo_pop = npis_data_ind[npis_data_ind['GeoID']==geo].population.values[0]
            median_case_preds.append(np.median(geo_pred_ind.PredictedDailyNewCases.values/(geo_pop/100000)))

        #Update X_ind
        pdb.set_trace()
        X_ind[:,12]=median_case_preds

        #Add cases and NPI sums
        #Check where 0
        #zind = np.argwhere(X_ind[:,12]==0)
        #X_ind[:,12][zind]=all_preds[zind]
        #diff = all_preds/X_ind[:,12]
        obj1 += np.sum(all_preds)
        obj2 += np.sum(prescr)

        #Update X_ind with case predictions
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
ip_costs['GeoID'] = np.where(ip_costs["RegionName"].isnull(),
                              ip_costs["CountryName"],
                              ip_costs["CountryName"] + ' / ' + ip_costs["RegionName"])
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

toolbox, creator, MU = setup_nsga3(NOBJ, NDIM, P, BOUND_LOW, BOUND_UP, CXPB, MUTPB,start_date,lookback_days,ip_costs)

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
