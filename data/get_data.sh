#Get the Oxford data
#wget https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv
#Get the regions that will be used
#https://github.com/leaf-ai/covid-xprize/blob/master/countries_regions.csv

###Preprocess
OXFORD_FILE=./OxCGRT_latest.csv
STATE_POPS=./populations/us_state_populations.csv
REGIONAL_POPS=./populations/regional_populations.csv
COUNTRY_POPS=./populations/country_populations.csv
GNI=./gni_per_capita_2019.csv
POP_DENSITY=./population_density.csv
MONTHLY_TEMP=./temp_1991_2016.csv
MOBILITY_DATA=/home/patrick/data/COVID19/xprize/Global_Mobility_Report.csv
CULTURAL_DESCRIPTORS=./cultural_descriptors-2015-08-16.csv
WORLD_AREAS=./iso3_to_world_area.csv
ADDITIONAL_XPRIZE_DATA=./additional_xprize_data/Additional_Context_Data_Global.csv
COUNTRY_REGIONS=./countries_regions.csv
OUTDIR=./
./preprocess.py --oxford_file $OXFORD_FILE --us_state_populations $STATE_POPS --regional_populations $REGIONAL_POPS --country_populations $COUNTRY_POPS --gross_net_income $GNI --population_density $POP_DENSITY --monthly_temperature $MONTHLY_TEMP --mobility_data $MOBILITY_DATA --cultural_descriptors $CULTURAL_DESCRIPTORS --world_areas $WORLD_AREAS --additional_xprize_data $ADDITIONAL_XPRIZE_DATA --country_regions $COUNTRY_REGIONS --outdir $OUTDIR
