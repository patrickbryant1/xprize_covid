#Predict using model

#4 days in August
#python predict.py -s 2020-08-01 -e 2020-08-04 -ip ../test_cases/2020-09-30_historical_ip.csv -o predictions_2020-08-01_2020-08-04.csv
ALL_HISTORICAL_IPS=../../data/historical_ip.csv
#Predit
python3 predict.py -s 2020-11-21 -e 2020-12-21 -ip $ALL_HISTORICAL_IPS -o predictions_2020-11-21_2020-12-21.csv


#Validate the model
FIP=/home/patrick/covid-xprize/covid_xprize/validation/data/future_ip.csv
#python3 predict.py -s 2020-12-23 -e 2021-01-01 -ip $FIP -o predictions_2020-12-23_2021-01-01.csv
