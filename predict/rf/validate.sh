
#Validate the model
#FIP=/home/patrick/covid-xprize/covid_xprize/validation/data/future_ip.csv
#python3 ../predictor_validation.py -s 2020-12-23 -e 2021-01-01 -ip $FIP - predictions_2020-12-23_2021-01-01.csv


ALL_HISTORICAL_IPS=../../data/hitorical_ip.csv
python3 ../predictor_validation.py -s 2020-11-20 -e 2020-12-11 -ip $ALL_HISTORICAL_IPS -sub predictions_2020-11-20_2020-12-11.csv
