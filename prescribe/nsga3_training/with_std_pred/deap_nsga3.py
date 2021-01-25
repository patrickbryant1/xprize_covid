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
import time

#Xprize predictor
# Suppress noisy Tensorflow debug logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.constraints import Constraint
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Lambda
from keras.models import Model
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
# Construct model
class Positive(Constraint):

    def __call__(self, w):
        return K.abs(w)

# Functions to be used for lambda layers in model
def _combine_r_and_d(x):
    r, d = x
    return r * (1. - d)

def construct_model(nb_context, nb_action, lstm_size=32, nb_lookback_days=21):
    '''Create the std predictor
    '''
    # Create context encoder
    context_input = Input(shape=(nb_lookback_days, nb_context),
                          name='context_input')
    x = LSTM(lstm_size, name='context_lstm')(context_input)
    context_output = Dense(units=1,
                           activation='softplus',
                           name='context_dense')(x)

    # Create action encoder
    # Every aspect is monotonic and nonnegative except final bias
    action_input = Input(shape=(nb_lookback_days, nb_action),
                         name='action_input')
    x = LSTM(units=lstm_size,
             kernel_constraint=Positive(),
             recurrent_constraint=Positive(),
             bias_constraint=Positive(),
             return_sequences=False,
             name='action_lstm')(action_input)
    action_output = Dense(units=1,
                          activation='sigmoid',
                          kernel_constraint=Positive(),
                          name='action_dense')(x)

    # Create prediction model
    model_output = Lambda(_combine_r_and_d, name='prediction')(
        [context_output, action_output])
    model = Model(inputs=[context_input, action_input],
                  outputs=[model_output])
    model.compile(loss='mae', optimizer='adam')

    return model

def load_model():
    '''Load the standard predictor
    '''
    global predictor
    # Load model weights
    nb_context = 1  # Only time series of new cases rate is used as context (PredictionRatio)
    nb_action = 12 #the NPI columns
    LSTM_SIZE = 32
    NB_LOOKBACK_DAYS = 21
    predictor = construct_model(nb_context=nb_context,
                                nb_action=nb_action,
                                lstm_size=LSTM_SIZE,
                                nb_lookback_days=NB_LOOKBACK_DAYS)
    # Fixed weights for the standard predictor.
    MODEL_WEIGHTS_FILE = './trained_model_weights.h5'
    predictor.load_weights(MODEL_WEIGHTS_FILE)

    return None

def load_inp_data(start_date,lookback_days,ip_costs):
    '''Load the input data for the standard predictor
    '''
    #Define global to use for all inds
    global X_prescr_inp, npis_data, ip_maxvals, ip_weights, X_pred_context_inp, X_pred_total_cases, X_pred_new_cases, populations

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
    #Preprocessed data
    DATA_FILE = '../../../data/adjusted_data.csv'
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

    #Get inp data for prescriptor
    data = data[DATA_COLUMNS]
    #Fill nans
    data[DATA_COLUMNS[4:]] = data[DATA_COLUMNS[4:]].fillna(0)
    data = data[(data.Date >= start_date) & (data.Date <= (pd.to_datetime(start_date, format='%Y-%m-%d') + np.timedelta64(lookback_days-1, 'D')))]

    #They predict percent change in new cases
    #This prediction ration is also the input to the next step
    # Compute percent change in new cases and deaths each day
    # Add column for proportion of population infected
    data['ProportionInfected'] = data['ConfirmedCases'] / data['population']
    # Compute number of new cases and deaths each day
    data['NewCases'] = data.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    # Replace negative values (which do not make sense for these columns) with 0
    data['NewCases'] = data['NewCases'].clip(lower=0)
    # Compute smoothed versions of new cases and deaths each day using a 7 day window
    data['SmoothNewCases'] = data.groupby('GeoID')['NewCases'].rolling(7, center=False).mean().fillna(0).reset_index(0, drop=True)
    # Compute percent change in new cases and deaths each day
    data['CaseRatio'] = data.groupby('GeoID').SmoothNewCases.pct_change(
    ).fillna(0).replace(np.inf, 0) + 1
    # Create column of value to predict
    data['PredictionRatio'] = data['CaseRatio'] / (1 - data['ProportionInfected'])
    #Normalize cases for prescriptor
    data['smoothed_cases']=data['smoothed_cases']/(data['population']/100000)

    #Format prescr inp data for prescriptor and pred for predictor
    ip_weights = []
    X_prescr_inp = []
    X_pred_context_inp = []
    X_pred_total_cases = []
    X_pred_new_cases = []
    populations = []
    for geo in data.GeoID.unique():
        geo_data = data[data['GeoID']==geo]
        X_geo= np.average(geo_data[DATA_COLUMNS[-12:]],axis=0)
        X_geo = np.append(X_geo,np.median(geo_data['smoothed_cases']))
        X_prescr_inp.append(X_geo)
        #Get ip weights
        ip_weights.append(ip_costs[ip_costs['GeoID']==geo][DATA_COLUMNS[-12:]].values[0])
        #Get input for predictor model. The context here is the PredictionRatio
        X_pred_context_inp.append(geo_data['PredictionRatio'].values)
        #Get the total and new cases needed for the predictor ratio conversions
        X_pred_total_cases.append(geo_data['ConfirmedCases'].values)
        X_pred_new_cases.append(geo_data['SmoothNewCases'].values)
        populations.append(geo_data.population.values[0])

    #Convert to array
    X_prescr_inp = np.array(X_prescr_inp)
    ip_weights = np.array(ip_weights)
    X_pred_context_inp = np.array(X_pred_context_inp)
    X_pred_total_cases = np.array(X_pred_total_cases)
    X_pred_new_cases = np.array(X_pred_new_cases)
    populations = np.array(populations)

    return None


def setup_nsga3(NOBJ, NDIM, P, BOUND_LOW, BOUND_UP, CXPB, MUTPB, start_date, lookback_days,ip_costs):
    #https://deap.readthedocs.io/en/master/examples/nsga3.html
    #Problem definition
    '''
    #Number of models in the pareto front
    '''
    H = 20 #factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))


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


def convert_ratios_to_total_cases(ratios, window_size, prev_new_cases, initial_total_cases,pop_sizes):
    '''Convert the ratios to get the case number output
    '''
    new_new_cases = []
    curr_total_cases = initial_total_cases
    for days_ahead in range(ratios.shape[1]): #each ratio will contain predictions for a region
        prev_pct_infected = curr_total_cases / pop_sizes

        new_cases = (ratios[:,days_ahead] * (1 - prev_pct_infected) - 1) * \
                    (window_size * np.mean(prev_new_cases[:,-window_size:],axis=1)) \
                    + prev_new_cases[:,-window_size]
        # new_cases can't be negative!
        new_cases[new_cases<0]=0
        # Which means total cases can't go down
        curr_total_cases += new_cases
        # Update prev_new_cases_list for next iteration of the loop
        prev_new_cases = np.concatenate((prev_new_cases,np.expand_dims(new_cases,axis=1)),axis=1)

    return prev_new_cases[:,-ratios.shape[1]:]

def roll_out_predictions(predictor, context_input, action_input, future_action_sequence, prev_confirmed_cases, prev_new_cases, pop_sizes):
    '''The predictions happen in steps of one day, why they have to be rolled out day by day.
    They also have to be converted to daily cases as some kind of ratios are predicted
    '''
    context_column = 'PredictionRatio'
    outcome_column = 'PredictionRatio'
    #Expand inp dims for NN
    context_input = np.expand_dims(context_input,axis=2)
    WINDOW_SIZE = 7
    nb_roll_out_days = future_action_sequence.shape[1]
    pred_output = np.zeros((future_action_sequence.shape[0],nb_roll_out_days))
    for d in range(nb_roll_out_days):

        #context input: (None, 21, 1)
        #action input  (None, 21, 12)
        pred = predictor.predict([context_input, action_input])
        pred_output[:,d] = pred[:,0]
        #Add the new action input according to the predictions
        action_input[:,-d+1,:] = future_action_sequence[:,d,:]
        context_input[:,-d+1,:] = pred

    #Convert to daily new cases
    # Compute number of new cases and deaths each day
    # Gather info to convert to total cases
    initial_total_cases = prev_confirmed_cases[:,-1] #Initial total cases
    pred_new_cases = convert_ratios_to_total_cases(pred_output,WINDOW_SIZE,prev_new_cases, initial_total_cases, pop_sizes)

    return pred_new_cases, pred_output



def evaluate_npis(individual):
    '''Evaluate the prescriptor by predicting the outcome using the
    pretrained predictor.
    '''
    #Get copy
    X_ind = np.copy(X_prescr_inp)
    X_context_ind = np.copy(X_pred_context_inp)
    X_total_cases_ind = np.copy(X_pred_total_cases)
    X_new_cases_ind = np.copy(X_pred_new_cases)
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
        prev_ip = X_ind[:,:12].copy()
        #Get cases in last period
        prev_cases = X_ind[:,12]
        #Multiply prev ip with the 2 prescr weight layers of the individual
        prescr = prev_ip*ip_weights*individual[:,0,0]*individual[:,0,1]
        #Add the case focus
        prescr += np.array([prev_cases]).T*individual[:,1,0]*individual[:,1,1]
        #Now the prescr can't really increase based only on the prescr
        #The cases will have to be high for the ips to increase
        #Perhaps this is not such a bad thing
        #Make sure the prescr don't exceeed the npi maxvals
        prescr = np.minimum(prescr,ip_maxvals)
        X_ind[:,:12]=prescr
        #Distribute the prescriptions in the forecast period
        #I choose to keep these stable over a three week period as changing them
        #on e.g. a daily or weekly basis in various degrees will not only make them
        #hard to follow but also confuse the public
        #Generate the predictions
        #Repeat the array for each region
        future_action_sequence = []
        previous_action_sequence = []
        for ri in range(prescr.shape[0]):
            future_action_sequence.append(np.tile(prescr[ri,:],[21,1]))
            previous_action_sequence.append(np.tile(prev_ip[ri,:],[21,1]))

        #time
        #tic = time.clock()
        pred_new_cases, pred_output = roll_out_predictions(predictor, X_context_ind, np.array(previous_action_sequence), np.array(future_action_sequence),X_total_cases_ind,X_new_cases_ind,populations)
        #toc = time.clock()
        #print(np.round(toc-tic,2))
        #preds have shape n_regions x forecast_days
        #Update X_context_ind
        X_context_ind = np.copy(pred_output)
        #Update new cases and toatl cases
        X_new_cases_ind = np.copy(pred_new_cases)
        #Get cumulative cases
        X_total_cases_ind = np.cumsum(np.concatenate((np.expand_dims(X_total_cases_ind[:,-1],axis=1),X_new_cases_ind),axis=1),axis=1)[:,1:]
        median_case_preds = np.median(pred_new_cases,axis=1)/(populations/100000)

        #Update X_ind with preds
        X_ind[:,12]=median_case_preds

        #Add cases and NPI sums
        #By adding over the total period - not only the final values will matter
        obj1 += np.sum(median_case_preds)
        obj2 += np.sum(prescr)

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
#Load the model input data
load_inp_data(start_date, lookback_days,ip_costs)

#Create the prescriptor setup
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
