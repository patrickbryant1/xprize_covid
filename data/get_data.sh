#Get the Oxford data
#wget https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv
#Get the google mobility data

###Preprocess
OXFORD_FILE=./OxCGRT_latest.csv
OUTDIR=./
./preprocess.py --oxford_file $OXFORD_FILE --outdir $OUTDIR
