POP=/home/patrick/results/COVID19/xprize/prescriptor/standard/population.npy
FRONT=/home/patrick/results/COVID19/xprize/prescriptor/standard/front.npy
SEL_FRONT='3,20,14,13,17,10,19,15,6,22'
OUTDIR=./prescr_weights/

./select_front.py --population $POP --total_front $FRONT --selected_front $SEL_FRONT --outdir $OUTDIR
