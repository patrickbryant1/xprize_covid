#Predict using model

#4 days in August
#python predict.py -s 2020-08-01 -e 2020-08-04 -ip ../test_cases/2020-09-30_historical_ip.csv -o predictions_2020-08-01_2020-08-04.csv
ALL_HISTORICAL_IPS=../../data/hitorical_ip.csv
#Predit March
python3 predict.py -s 2020-03-07 -e 2020-03-30 -ip $ALL_HISTORICAL_IPS -o predictions_2020-03-07_2020-03-31.csv
#Predit June
python3 predict.py -s 2020-06-07 -e 2020-06-30 -ip $ALL_HISTORICAL_IPS -o predictions_2020-06-07_2020-06-31.csv
#Predit November
python3 predict.py -s 2020-11-07 -e 2020-11-30 -ip $ALL_HISTORICAL_IPS -o predictions_2020-11-07_2020-11-31.csv
#1 month in January for India and Mexico
#python3 predict.py -s 2021-01-01 -e 2021-01-31 -ip ../test_cases/future_ip.csv -o predictions_2021-01-01_2021-01-31.csv
