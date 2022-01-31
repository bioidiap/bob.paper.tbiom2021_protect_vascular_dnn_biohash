RESULT_ROOT=../../../../../models


FVR=wld    
echo $FVR

RESULT_DIR=$RESULT_ROOT/$FVR
EvalTag=$FVR"_normal"
EVAL_DIR=evaluation
mkdir $EVAL_DIR

all_labels="L_Biohash=30,L_Biohash=50,L_Biohash=80,L_Biohash=100,L_Biohash=500,L_Biohash=1000"

bob bio metrics  $RESULT_DIR/AE/alpha_1/emb_100/verify/results/$FVR-AE-BHsh-{30,50,80,100,500,1000}/baseline/nom/nonorm/scores-{dev,eval} -l $EVAL_DIR/metrics_$EvalTag.txt

bob bio roc -e -v $RESULT_DIR/AE/alpha_1/emb_100/verify/results/$FVR-AE-BHsh-{30,50,80,100,500,1000}/baseline/nom/nonorm/scores-{dev,eval} -lg $all_labels -o $EVAL_DIR/ROC_$EvalTag.pdf --figsize 4.2,3.2

    
    
    RESULT_DIR=$RESULT_ROOT/$FVR
    EvalTag=$FVR"_stolen"
    EVAL_DIR=evaluation
    mkdir $EVAL_DIR
    
    #all_labels="$FVR+AE+Biohash,$FVR+PCA+Biohash,$FVR+Biohash"


bob bio metrics  $RESULT_DIR/AE/alpha_1/emb_100/verify/results/$FVR-AE-BHsh-{30,50,80,100,500,1000}-stolen/baseline/nom/nonorm/scores-{dev,eval} -l $EVAL_DIR/metrics_$EvalTag.txt

bob bio roc -e -v $RESULT_DIR/AE/alpha_1/emb_100/verify/results/$FVR-AE-BHsh-{30,50,80,100,500,1000}-stolen/baseline/nom/nonorm/scores-{dev,eval} -lg $all_labels -o $EVAL_DIR/ROC_$EvalTag.pdf --figsize 4.2,3.2
