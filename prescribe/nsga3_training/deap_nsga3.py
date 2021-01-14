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
import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simple RF model.''')

parser.add_argument('--adjusted_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')
parser.add_argument('--temp_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to data file with monthly temperatures.')
parser.add_argument('--ip_costs', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to data file with ip costs per region.')
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
def get_eval_inp_data(adjusted_data,train_days,ip_costs):
    '''Get the evaluation data
    '''
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
                        'GeoID',
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

    #Get only selected features
    sel = adjusted_data[selected_features]
    #Get all data
    X = [] #Input data to predictor
    regional_ipcosts = [] #Ip costs per region

    for geo in sel.GeoID.unique():

        #Get regional data
        country_region_data = sel[sel['GeoID']==geo]
        country_region_data = country_region_data.reset_index()
        #ip costs
        regional_ipcosts.append(ip_costs[ip_costs['GeoID']==geo][IP_MAX_VALUES.keys()].values[0])

        death_to_case_scale = country_region_data.loc[0,'death_to_case_scale']
        case_death_delay = country_region_data.loc[0,'case_death_delay']
        gross_net_income = country_region_data.loc[0,'gross_net_income']
        population_density = country_region_data.loc[0,'population_density']
        pdi = country_region_data.loc[0,'pdi'] #Power distance
        idv = country_region_data.loc[0, 'idv'] #Individualism
        mas = country_region_data.loc[0,'mas'] #Masculinity
        uai = country_region_data.loc[0,'uai'] #Uncertainty
        ltowvs = country_region_data.loc[0,'ltowvs'] #Long term orientation,  describes how every society has to maintain some links with its own past while dealing with the challenges of the present and future
        ivr = country_region_data.loc[0,'ivr'] #Indulgence, Relatively weak control is called “Indulgence” and relatively strong control is called “Restraint”.
        upop = country_region_data.loc[0,'Urban population (% of total population)']
        pop65 = country_region_data.loc[0,'Population ages 65 and above (% of total population)']
        gdp = country_region_data.loc[0,'GDP per capita (current US$)']
        obesity = country_region_data.loc[0,'Obesity Rate (%)']
        cancer = country_region_data.loc[0,'Cancer Rate (%)']
        smoking_deaths = country_region_data.loc[0,'Share of Deaths from Smoking (%)']
        pneumonia_dr = country_region_data.loc[0,'Pneumonia Death Rate (per 100K)']
        air_pollution_deaths = country_region_data.loc[0,'Share of Deaths from Air Pollution (%)']
        co2_emission = country_region_data.loc[0,'CO2 emissions (metric tons per capita)']
        air_transport = country_region_data.loc[0,'Air transport (# carrier departures worldwide)']
        population = country_region_data.loc[0,'population']
        country_region_data = country_region_data.drop(columns={'index','GeoID', 'death_to_case_scale', 'case_death_delay', 'gross_net_income','population_density','pdi', 'idv',
         'mas', 'uai', 'ltowvs', 'ivr','Urban population (% of total population)','Population ages 65 and above (% of total population)',
         'GDP per capita (current US$)', 'Obesity Rate (%)', 'Cancer Rate (%)', 'Share of Deaths from Smoking (%)', 'Pneumonia Death Rate (per 100K)',
         'Share of Deaths from Air Pollution (%)','CO2 emissions (metric tons per capita)', 'Air transport (# carrier departures worldwide)','population'})

        #Normalize the cases by 100'000 population
        country_region_data['smoothed_cases']=country_region_data['smoothed_cases']/(population/100000)
        country_region_data['cumulative_smoothed_cases']=country_region_data['cumulative_smoothed_cases']/(population/100000)
        #Get all features
        xi = np.array(country_region_data.loc[:train_days-1])
        case_medians = np.median(xi[:,12:14],axis=0)
        xi = np.average(xi,axis=0)
        xi[12:14]=case_medians

        #Add
        X.append(np.append(xi.flatten(),[death_to_case_scale,case_death_delay,gross_net_income,population_density,
                                        #period_change,
                                        pdi, idv, mas, uai, ltowvs, ivr,upop, pop65, gdp, obesity,
                                        cancer, smoking_deaths, pneumonia_dr, air_pollution_deaths, co2_emission,
                                        air_transport, population]))

    return np.array(X), np.array([*IP_MAX_VALUES.values()]), np.array(regional_ipcosts)


def load_model():
    '''Load the models
    '''
    #Make global
    global low_models
    global high_models
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


def setup_nsga3(NOBJ, NDIM, P, BOUND_LOW, BOUND_UP, CXPB, MUTPB):
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
    load_model()
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
    X_ind = np.copy(eval_inp_data)
    #Convert to array and reshape
    individual = np.array(individual)
    individual = np.reshape(individual,(12,NUM_WEIGHTS,NUM_LAYERS))
    #Prescribe and predict for n 21 day periods
    obj1 = 0 #Cumulative preds
    obj2 = 0 #Cumulative issued NPIs
    for n in range(4): #4 21 day periods = 3 months, which should be sufficient to observe substantial changes
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
        #Split into high and low
        high_i = np.argwhere(X_ind[:,12]>t)
        low_i = np.argwhere(X_ind[:,12]<=t)
        X_high = X_ind[high_i][:,0,:]
        X_low = X_ind[low_i][:,0,:]

        high_model_preds = []
        low_model_preds = []
        for mi in range(len(high_models)):
            high_model_preds.append(high_models[mi].predict(X_high))
            low_model_preds.append(low_models[mi].predict(X_low))
        #Convert to arrays
        high_model_preds = np.average(np.array(high_model_preds),axis=0)
        low_model_preds = np.average(np.array(low_model_preds),axis=0)
        #Concat
        X_ind = np.append(X_high, X_low,axis=0)
        all_preds = np.append(high_model_preds,low_model_preds)
        #Below 0 not allowed
        all_preds[all_preds<0]=0

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
monthly_temperature = pd.read_csv(args.temp_data[0])
ip_costs = pd.read_csv(args.ip_costs[0])
ip_costs['GeoID'] = ip_costs['CountryName'] + '__' + ip_costs['RegionName'].astype(str)
start_date = args.start_date[0]
train_days = args.train_days[0]
forecast_days = args.forecast_days[0]
threshold = args.threshold[0]
outdir = args.outdir[0]

#Get the input data
#Use only data from start date
adjusted_data = adjusted_data[adjusted_data['Date']>=start_date]
global eval_inp_data, ip_maxvals, ip_weights, t
eval_inp_data, ip_maxvals, ip_weights = get_eval_inp_data(adjusted_data, train_days, ip_costs)
t = threshold
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
toolbox, creator, MU = setup_nsga3(NOBJ, NDIM, P, BOUND_LOW, BOUND_UP, CXPB, MUTPB)

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
