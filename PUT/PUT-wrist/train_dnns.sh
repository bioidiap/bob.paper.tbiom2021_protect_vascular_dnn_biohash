root_dir=$(pwd)

for model in {wld,rlt,mc,img}
do
    cd $root_dir
    cd models
    cd $model
    cd AE/alpha_1/emb_100
    python train.py

done