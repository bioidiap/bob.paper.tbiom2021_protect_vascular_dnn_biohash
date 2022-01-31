root_dir=$(pwd)

for model in {wld,rlt,mc,img}
do
    cd $root_dir
    cd models
    cd $model
    cd AE/alpha_1/emb_100
    python train.py

done


# wld (for ablation study)
for alpha in {0..10}
do
    cd $root_dir
    cd models
    cd wld
    cd AE/alpha_$alpha/emb_100
    python train.py
done



# wld (for ablation study)
for L_emb in {50,200,500,1000}
do
    cd $root_dir
    cd models
    cd wld
    cd AE/alpha_1/emb_$L_emb
    python train.py
done