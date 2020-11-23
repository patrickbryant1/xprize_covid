#Get the Oxford data
#wget https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv
#Get the google mobility data

###Preprocess
OXFORD_FILE=./OxCGRT_latest.csv
STATE_POPS=./populations/us_state_populations.csv
REGIONAL_POPS=./populations/regional_populations.csv
COUNTRY_POPS=./populations/country_populations.csv
OUTDIR=./
./preprocess.py --oxford_file $OXFORD_FILE --us_state_populations $STATE_POPS --regional_populations $REGIONAL_POPS --country_populations $COUNTRY_POPS --outdir $OUTDIR
