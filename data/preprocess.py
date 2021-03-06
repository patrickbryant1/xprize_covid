#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Preprocess data for training.''')

parser.add_argument('--oxford_file', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to oxford data file.')
parser.add_argument('--us_state_populations', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to us_state_populations.')
parser.add_argument('--regional_populations', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to other regional populations (UK).')
parser.add_argument('--country_populations', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to country populations.')
parser.add_argument('--gross_net_income', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to country gross_net_income.')
parser.add_argument('--population_density', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to country population_density.')
parser.add_argument('--monthly_temperature', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to country monthly_temperature.')
parser.add_argument('--mobility_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to Google mobility data.')
parser.add_argument('--cultural_descriptors', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to Hofstede cultural descriptors.')
parser.add_argument('--world_areas', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to csv with ISO3 to world area.')
parser.add_argument('--additional_xprize_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to csv with additional data from xprize.')
parser.add_argument('--country_regions', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to csv with country regions from xprize.')
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

    delay_adjustments = {'BEN':7, 'BGD':11 ,'BRA':4,'CAN':7, 'COD':21, 'DEU':14, 'ECU':7,'EGY':7, 'ESP':11 ,'EST':10,'GAB':0, 'UK_ENG':17, 'ITA':7, 'JPN':14,'KWT':7,
                          'MOZ':7, 'MRT':0, 'PAN':7,'PRI':10, 'QAT':14, 'ROU':7,'RWA':0, 'SSD':7,'SYR':7, 'THA':14, 'TJK':0, 'TUR':7, 'USA':16, 'US_AL':10,
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

    #If the delay is still smaller than 7, the first 100 days of the curve are used
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
    #Some adjustments are done manually
    if region in [*delay_adjustments.keys()]:
        delay = delay_adjustments[region]
    #Scale index
    si = max(np.where(deaths>0)[0])
    scaling = cases[si-delay-1]/deaths[si]
    return death_maxi,case_maxi,delay,scaling,manual_adjust_necessary

def smooth_mobility(sel_mobility,whole_country_data):
    '''Smooth the mobility data
    '''
    #Mobility sectors
    mobility_sectors = ['retail_and_recreation_percent_change_from_baseline',
   'grocery_and_pharmacy_percent_change_from_baseline',
   'parks_percent_change_from_baseline',
   'transit_stations_percent_change_from_baseline',
   'workplaces_percent_change_from_baseline',
   'residential_percent_change_from_baseline']
    #Construct a 1-week sliding average to smooth the mobility data
    for sector in mobility_sectors:
        data = np.array(sel_mobility[sector])
        y = np.zeros(len(sel_mobility))
        for i in range(7,len(data)+1):
            #Check that there are no NaNs
            if np.isnan(data[i-7:i]).any():
                #If there are NaNs, loop through and replace with value from prev date
                for i_nan in range(i-7,i):
                    if np.isnan(data[i_nan]):
                        data[i_nan]=data[i_nan-1]
            y[i-1]=np.average(data[i-7:i])
        y[0:6] = y[6]
        sel_mobility.loc[sel_mobility.index,sector]=y



    #Join on date
    joined = pd.merge(whole_country_data,sel_mobility,left_on='Date',right_on='date',how='left')
    #Check that shape remains. There are problems with date joining
    if len(joined)!=len(whole_country_data):
        pdb.set_trace()
    mob_start = min(sel_mobility['date'])
    mob_end = max(sel_mobility['date'])
    #Replace the nan with the closest available data
    before_mob = joined[joined['Date']<mob_start].index
    after_mob = joined[joined['Date']>mob_end].index
    for sector in mobility_sectors:
        #Before mob data
        joined.at[before_mob,sector]=joined[joined['Date']==mob_start][sector].values[0]
        #After mob data
        joined.at[after_mob,sector]=joined[joined['Date']==mob_end][sector].values[0]
        #Check if still NaNs
        if len(joined[joined[sector].isna()])>1:
            nan_ind = joined[joined[sector].isna()].index
            min_nan = min(nan_ind)
            max_nan = max(nan_ind)
            mid_point = int(min_nan+(max_nan-min_nan)/2)
            #Set first half
            joined.at[min_nan:mid_point,sector] = joined.loc[min_nan-1,sector]
            #Set second half
            joined.at[mid_point:max_nan,sector] = joined.loc[max_nan+1,sector]

    return joined

def format_plot(region,rescaled_cases, case_maxi,cases,death_maxi,deaths,delay,scale,outname):
    '''Plot and format the plot
    '''
    #Create fig for vis
    fig,ax = plt.subplots(figsize=(6/2.54,4.5/2.54))

    #Add some noise to make log possible
    noise=0.01
    plt.plot(np.arange(cases.shape[0]),cases+noise,color='r',label='cases')
    plt.plot(np.arange(deaths.shape[0]),deaths+noise,color='k',label='deaths')

    plt.plot(np.arange(rescaled_cases.shape[0]),rescaled_cases+noise,color='b')
    plt.axvline(case_maxi,0,max(cases),color='r',linestyle='--', linewidth=1)
    plt.axvline(death_maxi,0,max(deaths),color='k',linestyle='--', linewidth=1)

    if scale=='log':
        plt.yscale('log')
    plt.xlabel('day')
    plt.title(region+'|'+str(delay))
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(outname, format='png', dpi=300)
    plt.close()

def parse_regions(oxford_data, us_state_populations, regional_populations, country_populations,
                gross_net_income,population_density,monthly_temperature,mobility_data,cultural_descriptors):
    '''Parse and encode all regions
    The ConfirmedCases column reports the total number of cases since
    the beginning of the epidemic for each country,region and day.
    From this number we can compute the daily change in confirmed cases.
    '''

    #Define country and regional indices
    oxford_data['Country_index']=0
    oxford_data['Region_index']=0
    oxford_data['smoothed_cases']=0
    oxford_data['cumulative_smoothed_cases']=0
    oxford_data['smoothed_deaths']=0
    oxford_data['rescaled_cases']=0
    oxford_data['cumulative_rescaled_cases']=0
    oxford_data['death_to_case_scale']=0
    oxford_data['case_death_delay']=0
    oxford_data['gross_net_income']=0
    oxford_data['population_density']=0
    oxford_data['Month']=oxford_data['Date'].dt.month
    oxford_data['monthly_temperature']=0

    temp_keys = {' Jan Average':1,' Feb Average':2,' Mar Average':3,' Apr Average':4,' May Average':5,' Jun Average':6,
                ' Jul Average':7,' Aug Average':8,' Sep Average':9,' Oct Average':10,' Nov Average':11, ' Dec Average':12}
    #Mobility sectors
    mobility_sectors = ['retail_and_recreation', 'grocery_and_pharmacy', 'parks',
                        'transit_stations', 'workplaces', 'residential']
    #Add sector
    for sector in mobility_sectors:
        oxford_data[sector]=0

    #Cultural descriptors
    cultural_keys = ['pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr']
    for cult_key in cultural_keys:
        oxford_data[cult_key]=0

    #Population
    oxford_data['population']=0

    #Unique countries
    country_codes = oxford_data['CountryCode'].unique()
    #No adjust regions
    no_adjust_regions = ['AFG','CAF','CHN','CHL','CIV','COD','COG','COM','GAB','DZA',
                        'LSO','MDG','MOZ','MWI','NAM','OMN','RWA','SAU','SEN','SMR',
                        'THA','TLS','TZA','YEM', 'US_VI', 'VNM', 'ZAF']

    manual_adjust_necessary = [] #Save the regions requiring manual adjustment
    manual_scaling = {'SWE':280} #took max(cases[-100:])/max(deaths[-100:])
    ci = 0 #Country index


    #Go through all countries and sub regions
    for cc in country_codes:
        ri = 0 #Region index
        country_data = oxford_data[oxford_data['CountryCode']==cc]
        #Get country total
        whole_country_data = country_data[country_data['RegionCode'].isna()]
        #Set index
        oxford_data.at[country_data.index,'Country_index']=ci
        #Get population
        population = country_populations[country_populations['Country Code']==cc]['2018'].values[0]
        oxford_data.at[country_data.index,'population']=population
        #Get income group
        country_gni = gross_net_income[gross_net_income['Country Code']==cc]['GNI per capita 2019 (USD)'].values[0]
        oxford_data.at[country_data.index,'gross_net_income']=country_gni
        #Get population density
        country_pop_density= population_density[population_density['Country Code']==cc]['Population density 2018'].values[0]
        oxford_data.at[country_data.index,'population_density']=country_pop_density
        #Get monthly_temperature
        #Regions not available - where a nearby region was used instead
        temp_conv_regions = {'ABW':'VEN','AIA':'PRI','BHS':'CUB','BMU':'CUB','CYM':'CUB','FLK':'ARG','GIB':'ESP',
                                'GMB':'GNB','GUM':'PHL','HKG':'CHN','KOR':'JPN','MAC':'CHN', 'MSR':'PRI','PCN':'NZL',
                                'PSE':'ISR','RKS':'SRB','SMR':'ITA','TCA':'CUB','TWN':'JPN','VGB':'DOM','VIR':'DOM'}
        if cc in [*temp_conv_regions.keys()]:
            country_temp= monthly_temperature[monthly_temperature['ISO3']==' '+temp_conv_regions[cc]]
        else:
            country_temp= monthly_temperature[monthly_temperature['ISO3']==' '+cc]

        for tkey in temp_keys:
            month_av =  np.round(np.average(country_temp[country_temp['Statistics']==tkey]['Temperature - (Celsius)']),1)
            oxford_data.at[country_data[country_data['Month']==temp_keys[tkey]].index,'monthly_temperature']=month_av

        #Get mobility
        country_mobility = mobility_data[mobility_data['country_region']==country_data['CountryName'].unique()[0]]
        whole_country_mobility = country_mobility[country_mobility['sub_region_1'].isna()] #Select whole country
        #For some countries the data is duplicated
        whole_country_mobility=whole_country_mobility.loc[whole_country_mobility['date'].drop_duplicates().index]

        if len(whole_country_mobility)<1:
            print('No mobility for', country_data['CountryName'].unique()[0])
        else:
            #Smooth mobility
            smoothed_mobility = smooth_mobility(whole_country_mobility,whole_country_data)
            #Add mobility
            for sector in mobility_sectors:
                #Duplicates have to be dropped since sometimes double dates are added
                oxford_data.at[whole_country_data.index,sector]=np.array(smoothed_mobility.loc[smoothed_mobility.index,sector+'_percent_change_from_baseline'])

        #Get cultural descriptors
        if cc in cultural_descriptors['ctr'].values:
            country_cultural_descriptors = cultural_descriptors[cultural_descriptors['ctr']==cc]
            for cult_key in cultural_keys:
                oxford_data.at[country_data.index,cult_key]=int(country_cultural_descriptors[cult_key].values[0])
        else:
            print('No cultural descriptors for',cc)

        #Smooth cases and deaths
        cases,deaths = smooth_cases_and_deaths(np.array(whole_country_data['ConfirmedCases']),np.array(whole_country_data['ConfirmedDeaths']))

        if max(deaths)<0.1:
            print('Less than 1 death for',cc)
            delay = 0
            scaling=0
        else:
            #Identify time lag between deaths and cases
            death_maxi,case_maxi,delay,scaling,manual_adjust_necessary = identify_case_death_lag(cases,deaths,cc,manual_adjust_necessary)

        #Recale
        #Recaled cases
        rescaled_cases = np.zeros(cases.shape) #Need to create a new array, otherwise the cases are also overwritten (?)
        rescaled_cases[:]=cases[:]
        if cc not in no_adjust_regions:
            if cc in [*manual_scaling.keys()]:
                scaling = manual_scaling[cc]
            rescaled_cases[:200-delay]=deaths[delay:200]*scaling

        #If the rescaled cases are smaller than the cases, set to cases
        if max(rescaled_cases) < max(cases):
            rescaled_cases = cases
        #Save smoothed cases
        oxford_data.at[whole_country_data.index,'smoothed_cases']=cases
        #Save cumulative smoothed cases
        oxford_data.at[whole_country_data.index,'cumulative_smoothed_cases']=np.cumsum(cases)
        #Save smoothed deaths
        oxford_data.at[whole_country_data.index,'smoothed_deaths']=deaths
        #Save the rescaled cases
        oxford_data.at[whole_country_data.index,'rescaled_cases']=rescaled_cases
        #Save cumulative rescaled cases
        oxford_data.at[whole_country_data.index,'cumulative_rescaled_cases']=np.cumsum(rescaled_cases)
        #Save the scaling
        oxford_data.at[whole_country_data.index,'death_to_case_scale']=scaling
        #Save the delay
        oxford_data.at[whole_country_data.index,'case_death_delay']=delay

        #Plot
        #format_plot(cc,rescaled_cases, case_maxi,cases,death_maxi,deaths,delay,'normal',outdir+'plots/'+cc+'.png')
        #format_plot(cc,rescaled_cases, case_maxi,cases,death_maxi,deaths,delay,'log',outdir+'plots/log/'+cc+'_log.png')
        #Get regions
        regions = country_data['RegionCode'].dropna().unique()
        #Check if regions
        if regions.shape[0]>0:
            for region in regions:
                country_region_data = country_data[country_data['RegionCode']==region]
                #Smooth cases and deaths
                cases,deaths = smooth_cases_and_deaths(np.array(country_region_data['ConfirmedCases']),np.array(country_region_data['ConfirmedDeaths']))
                #Set regional index
                oxford_data.at[country_region_data.index,'Region_index']=ri
                #Icrease ri
                ri+=1


                if max(deaths)<1:
                    print('Less than 1 death for',cc,region)
                    delay = 0
                    scaling= deaths[-1]/cases[-1]
                else:
                    #Identify time lag between deaths and cases
                    death_maxi,case_maxi,delay,scaling,manual_adjust_necessary = identify_case_death_lag(cases,deaths,region,manual_adjust_necessary)

                #Recaled cases
                rescaled_cases = cases
                if cc in [*manual_scaling.keys()]:
                    scaling = manual_scaling[cc]
                rescaled_cases[:200-delay]=deaths[delay:200]*scaling

                if region in no_adjust_regions:
                    rescaled_cases=cases

                #If the rescaled cases are smaller than the cases, set to cases
                if max(rescaled_cases) < max(cases):
                    rescaled_cases = cases

                #Save smoothed cases
                oxford_data.at[country_region_data.index,'smoothed_cases']=cases
                #Save cumulative smoothed cases
                oxford_data.at[country_region_data.index,'cumulative_smoothed_cases']=np.cumsum(cases)
                #Save smoothed deaths
                oxford_data.at[country_region_data.index,'smoothed_deaths']=deaths
                #Save rescaled cases
                oxford_data.at[country_region_data.index,'rescaled_cases']=rescaled_cases
                #Save cumulative rescaled cases
                oxford_data.at[country_region_data.index,'cumulative_rescaled_cases']=np.cumsum(rescaled_cases)
                #Save the scaling
                oxford_data.at[country_region_data.index,'death_to_case_scale']=scaling
                #Save the delay
                oxford_data.at[country_region_data.index,'case_death_delay']=delay
                #Get population
                if region in regional_populations['Region Code'].values:
                    oxford_data.at[country_region_data.index,'population']=regional_populations[regional_populations['Region Code']==region]['2019 population'].values[0]
                else:
                    region_name = country_region_data['RegionName'].unique()[0]
                    try:
                        oxford_data.at[country_region_data.index,'population']=us_state_populations[us_state_populations['State']==region_name]['Population'].values[0]
                    except:
                        print('No population')
                        pdb.set_trace()

                #Get mobility for subregion
                country_region_mobility = country_mobility[(country_mobility['sub_region_1']==country_region_data['RegionName'].unique()[0]) & (country_mobility['sub_region_2'].isna())] #Select country region
                if len(country_region_mobility)<1:
                    print('No mobility for', country_region_data['RegionName'].unique()[0])
                    print('Setting regional mobility to country mobility')
                    #Smooth mobility
                    smoothed_mobility = smooth_mobility(whole_country_mobility,country_region_data)
                    for sector in mobility_sectors:
                        oxford_data.at[country_region_data.index,sector]=np.array(smoothed_mobility[sector+'_percent_change_from_baseline'])
                else:
                    #Smooth mobility
                    smoothed_mobility = smooth_mobility(country_region_mobility,country_region_data)
                    #Add mobility
                    for sector in mobility_sectors:
                        try:
                            oxford_data.at[country_region_data.index,sector]=np.array(smoothed_mobility[sector+'_percent_change_from_baseline'])
                        except:
                            pdb.set_trace()
                #if country_region_data['RegionName'] in us_state_populations
                #Plot
                #format_plot(region,rescaled_cases, case_maxi,cases,death_maxi,deaths,delay, 'normal',outdir+'plots/'+cc+'_'+region+'.png')
                #format_plot(region,rescaled_cases, case_maxi,cases,death_maxi,deaths,delay,'log',outdir+'plots/log/'+cc+'_'+region+'_log.png')

        #Increase ci
        ci+=1
        plt.close()

    return oxford_data

def cluster_countries(oxford_data,outdir):
    '''Cluster the different epidemic curves and investigate their characteristics
    '''
    #Cumulativ case distribution - save total amount of reported cases
    cumulative_case_distribution = []
    #last 90 days of case data
    last90 = []
    #Unique countries
    country_codes = oxford_data['CountryCode'].unique()

    #Go through all countries and sub regions
    for cc in country_codes:
        country_data = oxford_data[oxford_data['CountryCode']==cc]
        #Get regions
        regions = country_data['RegionCode'].unique()
        #Check if regions
        for region in regions:
            country_region_data = country_data[country_data['RegionCode']==region]
            cumulative_case_distribution.append(country_region_data['cumulative_rescaled_cases'].values[-1])
            li = country_region_data['rescaled_cases'].index[-1]
            last90.append(np.array(country_region_data.loc[li-89:,'rescaled_cases']))



    #Plot the cumulative case distribution
    fig,ax = plt.subplots(figsize=(6/2.54,4.5/2.54))
    plt.hist(np.log10(np.array(cumulative_case_distribution)+1))
    plt.xlabel('log cumulative cases')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(outdir+'cumulative_case_distr.png', format='png', dpi=300)
    plt.close()

    #PCA
    last90 = np.array(last90)
    decomp90 = PCA().fit(last90)
    fig,ax = plt.subplots(figsize=(6/2.54,4.5/2.54))
    plt.bar(np.arange(90),decomp90.explained_variance_ratio_)
    plt.ylabel('explained_variance_ratio_')
    plt.xlabel('PC')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(outdir+'explained_variance.png', format='png', dpi=300)
    plt.close()
    #Transform
    transformed_last90 = decomp90.transform(last90)
    fig,ax = plt.subplots(figsize=(6/2.54,6/2.54))
    plt.scatter(transformed_last90[:,0],transformed_last90[:,1],s=0.1)
    plt.xlabel('PC1:'+str(np.round(decomp90.explained_variance_ratio_[0],2)))
    plt.ylabel('PC2:'+str(np.round(decomp90.explained_variance_ratio_[1],2)))
    plt.tight_layout()
    plt.savefig(outdir+'first2_PCs.png', format='png', dpi=300)
    plt.close()

    #Add to Oxford data
    oxford_data['PC1']=0
    oxford_data['PC2']=0
    #Go through all countries and sub regions
    cri=0
    for cc in country_codes:
        country_data = oxford_data[oxford_data['CountryCode']==cc]
        #Get regions
        regions = country_data['RegionCode'].unique()
        #Check if regions
        for region in regions:
            country_region_indices = country_data[country_data['RegionCode']==region].index
            oxford_data.at[country_region_indices,'PC1']=transformed_last90[cri,0]
            oxford_data.at[country_region_indices,'PC2']=transformed_last90[cri,1]
            cri+=1

    return oxford_data

def correlations_within_data(oxford_data, outdir):
    '''Analyze the correlations within the data
    '''

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
                        'smoothed_cases',
                        'cumulative_smoothed_cases',
                        'rescaled_cases',
                        'cumulative_rescaled_cases',
                        'death_to_case_scale',
                        'case_death_delay',
                        'gross_net_income',
                        'population_density',
                        'monthly_temperature',
                        'retail_and_recreation',
                        'grocery_and_pharmacy',
                        'parks',
                        'transit_stations',
                        'workplaces',
                        'residential',
                        'pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr',
                        'population']

    features_sel = np.array(oxford_data[selected_features])
    correlations = np.zeros((features_sel.shape[1],features_sel.shape[1]))
    for i in range(correlations.shape[0]):
        for j in range(i+1,correlations.shape[1]):
            print(i,j)
            try:
                correlations[i,j]=np.correlate(features_sel[:,i],features_sel[:,j])[0]
            except:
                continue

    plt.imshow(correlations)
    plt.xticks(ticks=range(correlations.shape[0]),labels=selected_features,rotation='vertical')
    plt.yticks(ticks=range(correlations.shape[1]),labels=selected_features)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


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
us_state_populations = pd.read_csv(args.us_state_populations[0])
regional_populations = pd.read_csv(args.regional_populations[0])
country_populations = pd.read_csv(args.country_populations[0])
gross_net_income = pd.read_csv(args.gross_net_income[0])
population_density = pd.read_csv(args.population_density[0])
monthly_temperature = pd.read_csv(args.monthly_temperature[0])
mobility_data = pd.read_csv(args.mobility_data[0],parse_dates=['date'])
cultural_descriptors = pd.read_csv(args.cultural_descriptors[0],sep=';')
#Replace the #NULL! with 0
cultural_descriptors=cultural_descriptors.replace('#NULL!',0)
world_areas = pd.read_csv(args.world_areas[0])
additional_xprize_data = pd.read_csv(args.additional_xprize_data[0])
country_regions = pd.read_csv(args.country_regions[0])
outdir = args.outdir[0]



#Get only the regions specified by the xprize team
oxford_data['GeoID'] = oxford_data['CountryName'] + '__' + oxford_data['RegionName'].astype(str)
country_regions['GeoID'] = country_regions['CountryName'] + '__' + country_regions['RegionName'].astype(str)
oxford_data = oxford_data[oxford_data['GeoID'].isin(country_regions['GeoID'].unique())]
oxford_data = oxford_data.reset_index()

#Save the NPIS for testing
oxford_data[['CountryName', 'RegionName',
           'Date', 'C1_School closing','C2_Workplace closing',
           'C3_Cancel public events','C4_Restrictions on gatherings',
            'C5_Close public transport','C6_Stay at home requirements',
            'C7_Restrictions on internal movement','C8_International travel controls',
            'H1_Public information campaigns','H2_Testing policy',
            'H3_Contact tracing','H6_Facial Coverings']].to_csv(outdir+'historical_ip.csv',index=False)

#Parse the data
try:
    oxford_data = pd.read_csv(outdir+'adjusted_data.csv',
    parse_dates=['Date'],
    encoding="ISO-8859-1",
    dtype={"RegionName": str,
           "RegionCode": str,
           "Country_index":int,
           "Region_index":int},
    error_bad_lines=False)



except:
    #Add world area to ocford data
    oxford_data = pd.merge(oxford_data,world_areas,on='CountryCode', how='left')
    #Add the additional_xprize_data
    oxford_data = pd.merge(oxford_data,additional_xprize_data,on='CountryCode', how='left')
    #Parse regions
    oxford_data = parse_regions(oxford_data, us_state_populations, regional_populations, country_populations,
                                gross_net_income,population_density,monthly_temperature,mobility_data,cultural_descriptors)
    oxford_data = oxford_data.fillna(0)
    #Look at country epidemic curves
    oxford_data = cluster_countries(oxford_data,outdir)
    #Save the adjusted data
    oxford_data.to_csv(outdir+'adjusted_data.csv')


#Analyze the auto correlations of the features
correlations_within_data(oxford_data, outdir)
#Get the dates for training
'''
The flags just tells the presence, while the features tell the strength?
['C1_School closing', 'C1_Flag', 'C2_Workplace closing', 'C2_Flag',
       'C3_Cancel public events', 'C3_Flag', 'C4_Restrictions on gatherings',
       'C4_Flag', 'C5_Close public transport', 'C5_Flag',
       'C6_Stay at home requirements', 'C6_Flag',
       'C7_Restrictions on internal movement', 'C7_Flag',
       'C8_International travel controls', 'E1_Income support', 'E1_Flag',
       'E2_Debt/contract relief', 'E3_Fiscal measures',
       'E4_International support', 'H1_Public information campaigns',
       'H1_Flag', 'H2_Testing policy', 'H3_Contact tracing',
       'H4_Emergency investment in healthcare', 'H5_Investment in vaccines',
       'H6_Facial Coverings', 'H6_Flag', 'M1_Wildcard', 'ConfirmedCases',
       'ConfirmedDeaths']

       Ij,t = [100*vj,t-0.5(Fj-fj,t)]/Nj
       Nj=max policy value
       vj=policy value
       Fj=Flag
       fj,t=recorded binary flag for that indicator

       The prescriptor will not have the flags as inputs however



Some indicators – C1-C7, E1, H1 and H6 – have an additional binary flag variable that
can be either 0 or 1. For C1-C7, H1 and H6 this corresponds to the geographic scope of the policy.
For E1, this flag variable corresponds to the sectoral scope of income support.

    ['StringencyIndex', 'StringencyIndexForDisplay',
       'StringencyLegacyIndex', 'StringencyLegacyIndexForDisplay',
       'GovernmentResponseIndex', 'GovernmentResponseIndexForDisplay',
       'ContainmentHealthIndex', 'ContainmentHealthIndexForDisplay',
       'EconomicSupportIndex', 'EconomicSupportIndexForDisplay']
Further, we produce two versions of each index. One with the raw calculated index values,
plus we produce a "display" version which will "smooth" over gaps in the last seven days,
populating each date with the last available "good" data point.

I should thus use the display indices.
'''
print('Preprocessing completed...')
pdb.set_trace()
