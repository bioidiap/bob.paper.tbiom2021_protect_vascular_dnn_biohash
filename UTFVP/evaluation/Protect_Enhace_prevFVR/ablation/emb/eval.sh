RESULT_ROOT=../../../../../models


FVR=wld    
echo $FVR

RESULT_DIR=$RESULT_ROOT/$FVR
EvalTag=$FVR"_normal"
EVAL_DIR=evaluation
mkdir $EVAL_DIR

all_labels="L_embedding=50,L_embedding=100,L_embedding=200,L_embedding=500,L_embedding=1000"

bob bio metrics  $RESULT_DIR/AE/alpha_1/emb_{50,100,200,500,1000}/verify/results/$FVR-AE-BHsh-100/baseline/nom/nonorm/scores-{dev,eval} -l $EVAL_DIR/metrics_$EvalTag.txt

bob bio roc -e -v $RESULT_DIR/AE/alpha_1/emb_{50,100,200,500,1000}/verify/results/$FVR-AE-BHsh-100/baseline/nom/nonorm/scores-{dev,eval} -lg $all_labels -o $EVAL_DIR/ROC_$EvalTag.pdf --figsize 4.2,3.2

    
    
    RESULT_DIR=$RESULT_ROOT/$FVR
    EvalTag=$FVR"_stolen"
    EVAL_DIR=evaluation
    mkdir $EVAL_DIR
    
    #all_labels="$FVR+AE+Biohash,$FVR+PCA+Biohash,$FVR+Biohash"


bob bio metrics  $RESULT_DIR/AE/alpha_1/emb_{50,100,200,500,1000}/verify/results/$FVR-AE-BHsh-100-stolen/baseline/nom/nonorm/scores-{dev,eval} -l $EVAL_DIR/metrics_$EvalTag.txt

bob bio roc -e -v $RESULT_DIR/AE/alpha_1/emb_{50,100,200,500,1000}/verify/results/$FVR-AE-BHsh-100-stolen/baseline/nom/nonorm/scores-{dev,eval} -lg $all_labels -o $EVAL_DIR/ROC_$EvalTag.pdf --figsize 4.2,3.2
