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
    verify.py $model-BHsh-500.py -vvv
    verify.py $model-BHsh-500-stolen.py -vvv
    verify.py $model-BHsh-1000.py -vvv
    verify.py $model-BHsh-1000-stolen.py -vvv
done


# img
alpha=1
cd $root_dir
cd models
cd img/AE
cd AE/alpha_$alpha/emb_100/verify
verify.py $model-AE-BHsh-100.py -vvv
verify.py $model-AE-BHsh-100-stolen.py -vvv
