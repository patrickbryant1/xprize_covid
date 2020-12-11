#Get the Oxford data
#wget https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv
#Get the google mobility data

###Preprocess
OXFORD_FILE=../OxCGRT_latest.csv
REGIONAL_POPS=../populations/regional_populations.csv
COUNTRY_POPS=../populations/country_populations.csv
OUTDIR=./
./look_at_brazil.py --oxford_file $OXFORD_FILE --regional_populations $REGIONAL_POPS --country_populations $COUNTRY_POPS --outdir $OUTDIR
