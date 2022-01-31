RESULT_ROOT=../../../models


for FVR in {wld,mc,rlt}
do 
    
    echo $FVR
    
    test_scenario=100
    RESULT_DIR=$RESULT_ROOT/$FVR
    EvalTag=$FVR"_normal"
    EVAL_DIR=evaluation
    mkdir $EVAL_DIR
    
    all_labels="$FVR+AE+Biohash,$FVR+PCA+Biohash,$FVR+Biohash,$FVR"

    bob bio metrics  $RESULT_DIR/{AE/alpha_1/emb_100/verify/results/$FVR-AE-BHsh-$test_scenario,PCA/emb_100/results/$FVR-PCA-BHsh-$test_scenario,Baseline/results/$FVR-BHsh-$test_scenario,Baseline/results/$FVR}/baseline/nom/nonorm/scores-{dev,eval} -l $EVAL_DIR/metrics_$EvalTag.txt

    bob bio roc -e -v $RESULT_DIR/{AE/alpha_1/emb_100/verify/results/$FVR-AE-BHsh-$test_scenario,PCA/emb_100/results/$FVR-PCA-BHsh-$test_scenario,Baseline/results/$FVR-BHsh-$test_scenario,Baseline/results/$FVR}/baseline/nom/nonorm/scores-{dev,eval} -lg $all_labels -o $EVAL_DIR/ROC_$EvalTag.pdf --figsize 4.75,3.5
    
    
    test_scenario=100-stolen
    RESULT_DIR=$RESULT_ROOT/$FVR
    EvalTag=$FVR"_stolen"
    EVAL_DIR=evaluation
    mkdir $EVAL_DIR
    
    all_labels="$FVR+AE+Biohash,$FVR+PCA+Biohash,$FVR+Biohash"

    bob bio metrics  $RESULT_DIR/{AE/alpha_1/emb_100/verify/results/$FVR-AE-BHsh-$test_scenario,PCA/emb_100/results/$FVR-PCA-BHsh-$test_scenario,Baseline/results/$FVR-BHsh-$test_scenario}/baseline/nom/nonorm/scores-{dev,eval} -l $EVAL_DIR/metrics_$EvalTag.txt

    bob bio roc -e -v $RESULT_DIR/{AE/alpha_1/emb_100/verify/results/$FVR-AE-BHsh-$test_scenario,PCA/emb_100/results/$FVR-PCA-BHsh-$test_scenario,Baseline/results/$FVR-BHsh-$test_scenario}/baseline/nom/nonorm/scores-{dev,eval} -lg $all_labels -o $EVAL_DIR/ROC_$EvalTag.pdf --figsize 4.75,3.5

done
