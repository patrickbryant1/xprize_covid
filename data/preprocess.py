#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Preprocess data for training.''')

parser.add_argument('--oxford_file', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to oxford data file.')

parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


def smooth_cases_and_deaths(cases,deaths):
    '''Calculate 7-day rolling averages
    '''

    #Smooth cases
    #Replace nans with zeros
    cases[np.isnan(cases)]=0
    #Calculate the daily cases
    cases[1:]=cases[1:]-cases[0:-1]
    #Replace negatives with zeros
    cases[np.where(cases<0)]=0
    sm_cases = np.zeros(len(cases))
    for i in range(7,len(cases)+1):
        sm_cases[i-1]=np.average(cases[i-7:i])
    sm_cases[0:6] = sm_cases[6]

    #Smooth deaths
    deaths[np.isnan(deaths)]=0
    #Calculate the daily deaths
    deaths[1:]=deaths[1:]-deaths[0:-1]
    #Replace negatives with zeros
    deaths[np.where(deaths<0)]=0
    sm_deaths = np.zeros(len(deaths))
    for i in range(7,len(deaths)+1):
        sm_deaths[i-1]=np.average(deaths[i-7:i])
    sm_deaths[0:6] = sm_deaths[6]

    return sm_cases, sm_deaths

def identify_case_death_lag(cases,deaths,region,manual_adjust_necessary):
    '''Find the time lag between deaths and cases
    I do this by comparing the deaths and cases surrounding the peaks and working backwards to
    identify when these overlap.
    '''

    delay_adjustments = {'BEN':7, 'BRA':4,'CAN':7, 'COD':21, 'DEU':14, 'ECU':7, 'GAB':0, 'UK_ENG':17, 'ITA':7, 'JPN':7,'KWT':7,
                          'MOZ':7, 'MRT':0, 'PRI':10, 'ROU':7,'RWA':0, 'SYR':7, 'THA':14, 'TJK':0, 'TUR':7, 'USA':16, 'US_AL':10,
                          'US_AZ':21, 'US_CA':17, 'US_CO':17, 'US_GA':19, 'US_HI':17, 'US_IL':17, 'US_MA':17, 'US_MD':17,
                          'US_MI':17, 'US_MO':17, 'US_NV':17, 'US_OH':21, 'US_OR':17 ,'US_PA':17, 'US_SC':17, 'US_TN':21, 'US_TX':17, 'US_UT':17,'US_VA':17}


    case_maxi = np.where(cases==max(cases[-100:]))[0][-1]
    death_maxi = np.where(deaths==max(deaths[-100:]))[0][-1]
    delay = death_maxi-case_maxi
    #If the delay is smaller than 7, the second peak probably hasn't been
    #reached yet, why the "valleys" are assessed instead
    if delay <7:
        case_mini = np.where(cases==min(cases[-200:]))[0][0]
        death_mini = np.where(deaths==min(deaths[-200:]))[0][0]
        delay = death_mini-case_mini


    #If the delay is still smaller than 7, the first 200 days of the curve are used
    if delay <7:
        case_maxi = np.where(cases==max(cases[0:100]))[0][0]
        death_maxi = np.where(deaths==max(deaths[0:100]))[0][0]
        delay = death_maxi-case_maxi



    if delay<0:
        print('Delay only', delay, 'for',region)
        print('Setting delay to 0')
        delay=0

    if delay > 35:
        manual_adjust_necessary.append(region)
        delay=0
    if region in [*delay_adjustments.keys()]:
        delay = delay_adjustments[region]
    #Scale index
    si = max(np.where(deaths>0)[0])
    scaling = cases[si-delay-1]/deaths[si]
    return death_maxi,case_maxi,delay,scaling,manual_adjust_necessary



def parse_regions(oxford_data):
    '''Parse and encode all regions
    The ConfirmedCases column reports the total number of cases since
    the beginning of the epidemic for each country,region and day.
    From this number we can compute the daily change in confirmed cases.
    '''

    #Define country and regional indices
    oxford_data['Country_index']=0
    oxford_data['Region_index']=0
    oxford_data['rescaled_cases']=0
    country_codes = oxford_data['CountryCode'].unique()
    no_adjust_regions = ['AFG','CAF','CHN','CHL','CIV','COD','COG','COM','GAB','DZA','LSO','MDG','MOZ','MWI','NAM','OMN','RWA','SAU','SEN','SMR','THA','TZA','YEM', 'US_VI', 'VNM', 'ZAF']#No adjust regions
    manual_adjust_necessary = [] #Save the regions requiring manual adjustment
    ci = 0 #Country index
    for cc in country_codes:
        #Create fig for vis
        fig,ax = plt.subplots(figsize=(6/2.54,4.5/2.54))
        ri = 0 #Region index
        country_data = oxford_data[oxford_data['CountryCode']==cc]
        #Set index
        oxford_data.at[country_data.index,'Country_index']=ci

        #Plot total
        whole_country_data = country_data[country_data['RegionCode'].isna()]
        #Smooth cases and deaths
        cases,deaths = smooth_cases_and_deaths(np.array(whole_country_data['ConfirmedCases']),np.array(whole_country_data['ConfirmedDeaths']))
        #Plot
        #Add some noise to make log possible
        noise=0.01
        plt.plot(np.arange(cases.shape[0]),cases+noise,color='r',label='cases')
        plt.plot(np.arange(deaths.shape[0]),deaths+noise,color='k',label='deaths')

        if max(deaths)<1:
            print('Less than 1 death for',cc)
            delay = 0
            scaling=0
        else:
            #Identify time lag between deaths and cases
            death_maxi,case_maxi,delay,scaling,manual_adjust_necessary = identify_case_death_lag(cases,deaths,cc,manual_adjust_necessary)

        #Recale
        #Recaled cases
        rescaled_cases = cases
        if cc not in no_adjust_regions:
            if delay ==0:
                rescaled_cases=deaths*scaling
            else:
                rescaled_cases[:-delay]=deaths[delay:]*scaling
        #Save the rescaled cases
        oxford_data.at[whole_country_data.index,'rescaled_cases']=rescaled_cases
        #Plot
        plt.plot(np.arange(rescaled_cases.shape[0]),rescaled_cases+noise,color='b')
        plt.axvline(case_maxi,0,max(cases),color='r',linestyle='--', linewidth=1)
        plt.axvline(death_maxi,0,max(deaths),color='k',linestyle='--', linewidth=1)

        #plt.yscale('log')
        plt.xlabel('day')
        plt.title(cc+'|'+str(delay))
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(outdir+'plots/'+cc+'.png', format='png', dpi=300)
        # if cc in manual_adjust_necessary:
        #     plt.show()
        plt.close()

        #Get regions
        regions = country_data['RegionCode'].dropna().unique()
        #Check if regions
        if regions.shape[0]>0:
            for region in regions:
                #Create fig for vis
                fig,ax = plt.subplots(figsize=(6/2.54,4.5/2.54))
                country_region_data = country_data[country_data['RegionCode']==region]
                #Smooth cases and deaths
                cases,deaths = smooth_cases_and_deaths(np.array(country_region_data['ConfirmedCases']),np.array(country_region_data['ConfirmedDeaths']))
                #Set regional index
                oxford_data.at[country_region_data.index,'Country_index']=ri
                #Icrease ri
                ri+=1
                #Plot
                plt.plot(np.arange(cases.shape[0]),cases+noise,color='r',linewidth=1)
                plt.plot(np.arange(deaths.shape[0]),deaths+noise,color='k', linewidth=1)

                if max(deaths)<1:
                    print('Less than 1 death for',cc,region)
                    delay = 0
                    scaling= deaths[-1]/cases[-1]
                else:
                    #Identify time lag between deaths and cases
                    death_maxi,case_maxi,delay,scaling,manual_adjust_necessary = identify_case_death_lag(cases,deaths,region,manual_adjust_necessary)

                #Recaled cases
                rescaled_cases = cases
                if delay ==0:
                    rescaled_cases=deaths*scaling
                else:
                    rescaled_cases[:-delay]=deaths[delay:]*scaling

                if region in no_adjust_regions:
                    rescaled_cases=cases

                #Save rescaled cases
                oxford_data.at[country_region_data.index,'rescaled_cases']=rescaled_cases
                plt.plot(np.arange(rescaled_cases.shape[0]),rescaled_cases+noise,color='b')
                plt.axvline(case_maxi,0,max(cases),color='r',linestyle='--', linewidth=1)
                plt.axvline(death_maxi,0,max(deaths),color='k',linestyle='--', linewidth=1)

                #plt.yscale('log')
                plt.xlabel('day')
                plt.title(cc+'_'+region+'|'+str(delay))
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(outdir+'plots/'+cc+'_'+region+'.png', format='png', dpi=300)
                # if region in manual_adjust_necessary:
                #     plt.show()
                plt.close()


        #Increase ci
        ci+=1

    return oxford_data
    pdb.set_trace()
#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()
oxford_data = pd.read_csv(args.oxford_file[0],
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)
outdir = args.outdir[0]


oxford_data = parse_regions(oxford_data)
pdb.set_trace()
