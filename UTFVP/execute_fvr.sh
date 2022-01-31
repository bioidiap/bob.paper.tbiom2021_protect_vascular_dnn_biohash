root_dir=$(pwd)

for model in {wld,rlt,mc}
do
    # AE
    cd $root_dir
    cd models
    cd $model
    cd AE/alpha_1/emb_100/verify
    verify.py $model-AE-BHsh-100.py -vvv
    verify.py $model-AE-BHsh-100-stolen.py -vvv


    # Baseline
    cd $root_dir
    cd models
    cd $model
    cd Baseline
    verify.py $model.py -vvv
    verify.py $model-BHsh-100.py -vvv
    verify.py $model-BHsh-100-stolen.py -vvv

    # PCA
    cd $root_dir
    cd models
    cd $model
    cd PCA
    python PCA.py
    verify.py $model-PCA-BHsh-100.py -vvv
    verify.py $model-PCA-BHsh-100-stolen.py -vvv
done


# img
model=img
cd $root_dir
cd models
cd $model
cd AE/alpha_$alpha/emb_100/verify
verify.py $model-AE-BHsh-100.py -vvv
verify.py $model-AE-BHsh-100-stolen.py -vvv


# wld - ablation (effect of L_biohash) 
model=wld
for l_BHsh in {50,200}
do
    cd $root_dir
    cd models
    cd $model
    cd AE/alpha_1/emb_100/verify
    verify.py $model-AE-BHsh-$l_BHsh.py -vvv
    verify.py $model-AE-BHsh-$l_BHsh-stolen.py -vvv
done

# wld - ablation (effect of L_emb) 
model=wld
for L_emb in {200,500,1000}
do
    cd $root_dir
    cd models
    cd $model
    cd AE/alpha_1/emb_$L_emb/verify
    verify.py $model-AE-BHsh-100.py -vvv
    verify.py $model-AE-BHsh-100-stolen.py -vvv
done

# wld - ablation (effect of alpha)
for alpha in {0..10}
do
    cd $root_dir
    cd models
    cd $model
    cd AE/alpha_$alpha/emb_100/verify
    verify.py $model-AE-BHsh-100.py -vvv
    verify.py $model-AE-BHsh-100-stolen.py -vvv
done



# wld - (effect of L_biohash and L_emb) 
model=wld
for L_emb in {50,100,200,500}
do
    for l_BHsh in {25,50,75,100,200,500,1000}
    do
    cd $root_dir
    cd models
    cd $model
    cd AE/alpha_1/emb_$L_emb/verify
    verify.py $model-AE-BHsh-$l_BHsh.py -vvv
    verify.py $model-AE-BHsh-$l_BHsh-stolen.py -vvv
    done
done