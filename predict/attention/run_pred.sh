#Predict using model

ALL_HISTORICAL_IPS=../../data/hitorical_ip.csv
#Predit March
python3 predict.py -s 2020-03-07 -e 2020-03-30 -ip $ALL_HISTORICAL_IPS -o predictions_2020-03-07_2020-03-31.csv
#Predit June
python3 predict.py -s 2020-06-07 -e 2020-06-30 -ip $ALL_HISTORICAL_IPS -o predictions_2020-06-07_2020-06-31.csv
#Predit November
python3 predict.py -s 2020-11-07 -e 2020-11-30 -ip $ALL_HISTORICAL_IPS -o predictions_2020-11-07_2020-11-31.csv
