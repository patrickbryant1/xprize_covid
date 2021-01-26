POP=/home/patrick/results/COVID19/xprize/prescriptor/standard/population.npy
FRONT=/home/patrick/results/COVID19/xprize/prescriptor/standard/front.npy
SEL_FRONT='0,20,12,23,11,17,18,19,8,22'
OUTDIR=./prescr_weights/

./select_front.py --population $POP --total_front $FRONT --selected_front $SEL_FRONT --outdir $OUTDIR
