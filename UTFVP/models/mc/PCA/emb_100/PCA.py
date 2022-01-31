print("start the code")
import numpy as np
    
all_feats = np.load('/idiap/temp/hotroshi/paper/UTFVP/data/all_mc_aug_feats.npy')
#all_imgs = np.load('/idiap/temp/hotroshi/paper/data/all_imgs.npy')
######### Keras
#all_imgs  = np.reshape(all_imgs, (all_imgs.shape[0], all_imgs.shape[1], all_imgs.shape[2],1))
all_feats = np.reshape(all_feats,(all_feats.shape[0], all_feats.shape[1], all_feats.shape[2],1))

import pickle
#with open('/idiap/temp/hotroshi/paper/data/all_client_ids_aug.pkl','rb') as f:
#    all_client_ids=pickle.load(f)
    
    
print("data is ready")

import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
pca.fit(np.reshape(all_feats,(all_feats.shape[0],-1)))

with open('pca.pkl','wb') as f:
    pickle.dump(pca,f)