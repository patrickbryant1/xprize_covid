
#Validate the model
FIP=/home/patrick/covid-xprize/covid_xprize/validation/data/future_ip.csv
python3 ../predictor_validation.py -s 2020-12-23 -e 2021-01-01 -ip $FIP -sub predictions_2020-12-23_2021-01-01.csv
