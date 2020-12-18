#Predict using model

ALL_HISTORICAL_IPS=../../../data/historical_ip.csv
#Predit
python3 predict.py -s 2020-11-20 -e 2020-12-16 -ip $ALL_HISTORICAL_IPS -o predictions_2020-11-20_2020-12-16.csv
