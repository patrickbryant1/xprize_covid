
PARAMSFILE=/home/patrick/results/COVID19/xprize/CNN/20201204/params.txt
PARAMSORDER='filters,dilation_rate,kernel_size,lr,num_convolutional_layers'
RESULTSDIR=/home/patrick/results/COVID19/xprize/CNN/20201204/
OUTDIR=./
./eval_cnn.py --params_file $PARAMSFILE --params_order $PARAMSORDER --results_dir $RESULTSDIR --outdir $OUTDIR
