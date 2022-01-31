RESULT_ROOT=../../../models


FVR=wld    
echo $FVR

RESULT_DIR=$RESULT_ROOT/$FVR
EvalTag=$FVR"_normal"
EVAL_DIR=evaluation
mkdir $EVAL_DIR

all_labels="\$\alpha=0.0\$,\$\alpha=0.1\$,\$\alpha=0.2\$,\$\alpha=0.3\$,\$\alpha=0.4\$,\$\alpha=0.5\$,\$\alpha=0.6\$,\$\alpha=0.7\$,\$\alpha=0.8\$,\$\alpha=0.9\$,\$\alpha=1.0\$"

bob bio metrics  $RESULT_DIR/AE/alpha_{0,1,2,3,4,5,6,7,8,9,10}/emb_100/verify/results/$FVR-AE-BHsh-100/baseline/nom/nonorm/scores-{dev,eval} -l $EVAL_DIR/metrics_$EvalTag.txt

bob bio roc -e -v $RESULT_DIR/AE/alpha_{0,1,2,3,4,5,6,7,8,9,10}/emb_100/verify/results/$FVR-AE-BHsh-100/baseline/nom/nonorm/scores-{dev,eval} -lg $all_labels -o $EVAL_DIR/ROC_$EvalTag.pdf --figsize 4.2,3.2

    
    
    RESULT_DIR=$RESULT_ROOT/$FVR
    EvalTag=$FVR"_stolen"
    EVAL_DIR=evaluation
    mkdir $EVAL_DIR
    
    #all_labels="$FVR+AE+Biohash,$FVR+PCA+Biohash,$FVR+Biohash"


bob bio metrics  $RESULT_DIR/AE/alpha_{0,1,2,3,4,5,6,7,8,9,10}/emb_100/verify/results/$FVR-AE-BHsh-100-stolen/baseline/nom/nonorm/scores-{dev,eval} -l $EVAL_DIR/metrics_$EvalTag.txt

bob bio roc -e -v $RESULT_DIR/AE/alpha_{0,1,2,3,4,5,6,7,8,9,10}/emb_100/verify/results/$FVR-AE-BHsh-100-stolen/baseline/nom/nonorm/scores-{dev,eval} -lg $all_labels -o $EVAL_DIR/ROC_$EvalTag.pdf --figsize 4.2,3.2
