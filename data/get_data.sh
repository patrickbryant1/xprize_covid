#Get the Oxford data
#wget https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv
#Get the google mobility data

###Preprocess
OXFORD_FILE=./OxCGRT_latest.csv
STATE_POPS=./populations/us_state_populations.csv
REGIONAL_POPS=./populations/regional_populations.csv
COUNTRY_POPS=./populations/country_populations.csv
GNI=./gni_per_capita_2019.csv
POP_DENSITY=./population_density.csv
MONTHLY_TEMP=./temp_1991_2016.csv
OUTDIR=./
./preprocess.py --oxford_file $OXFORD_FILE --us_state_populations $STATE_POPS --regional_populations $REGIONAL_POPS --country_populations $COUNTRY_POPS --gross_net_income $GNI --population_density $POP_DENSITY --monthly_temperature $MONTHLY_TEMP --outdir $OUTDIR
