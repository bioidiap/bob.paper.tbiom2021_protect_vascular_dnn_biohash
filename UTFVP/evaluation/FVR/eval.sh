RESULT_ROOT=../../../models



RESULT_DIR=$RESULT_ROOT/$FVR
EvalTag="_normal"
EVAL_DIR=evaluation
mkdir $EVAL_DIR

all_labels="img+AE+Biohash,wld+AE+Biohash,rlt+AE+Biohash,mc+AE+Biohash"

bob bio metrics  $RESULT_ROOT/{img/AE/alpha_1/emb_100/verify/results/img-AE-BHsh-100,wld/AE/alpha_1/emb_100/verify/results/wld-AE-BHsh-100,rlt/AE/alpha_1/emb_100/verify/results/rlt-AE-BHsh-100,mc/AE/alpha_1/emb_100/verify/results/mc-AE-BHsh-100}/baseline/nom/nonorm/scores-{dev,eval} -l $EVAL_DIR/metrics_$EvalTag.txt

bob bio roc -e -v $RESULT_ROOT/{img/AE/alpha_1/emb_100/verify/results/img-AE-BHsh-100,wld/AE/alpha_1/emb_100/verify/results/wld-AE-BHsh-100,rlt/AE/alpha_1/emb_100/verify/results/rlt-AE-BHsh-100,mc/AE/alpha_1/emb_100/verify/results/mc-AE-BHsh-100}/baseline/nom/nonorm/scores-{dev,eval}  -lg $all_labels -o $EVAL_DIR/ROC_$EvalTag.pdf --figsize 4.2,3.2

    
    
RESULT_DIR=$RESULT_ROOT/$FVR
EvalTag="_stolen"
EVAL_DIR=evaluation
mkdir $EVAL_DIR
    

bob bio metrics  $RESULT_ROOT/{img/AE/alpha_1/emb_100/verify/results/img-AE-BHsh-100,wld/AE/alpha_1/emb_100/verify/results/wld-AE-BHsh-100,rlt/AE/alpha_1/emb_100/verify/results/rlt-AE-BHsh-100,mc/AE/alpha_1/emb_100/verify/results/mc-AE-BHsh-100}-stolen/baseline/nom/nonorm/scores-{dev,eval}  -l $EVAL_DIR/metrics_$EvalTag.txt

bob bio roc -e -v $RESULT_ROOT/{img/AE/alpha_1/emb_100/verify/results/img-AE-BHsh-100,wld/AE/alpha_1/emb_100/verify/results/wld-AE-BHsh-100,rlt/AE/alpha_1/emb_100/verify/results/rlt-AE-BHsh-100,mc/AE/alpha_1/emb_100/verify/results/mc-AE-BHsh-100}-stolen/baseline/nom/nonorm/scores-{dev,eval} -lg $all_labels -o $EVAL_DIR/ROC_$EvalTag.pdf --figsize 4.2,3.2
