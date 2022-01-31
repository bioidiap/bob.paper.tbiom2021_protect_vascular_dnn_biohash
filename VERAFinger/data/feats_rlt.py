print("start the code")
import numpy as np
from bob.bio.vein.preprocessor import NoCrop, TomesLeeMask, HuangNormalization, \
    NoFilter, Preprocessor

preprocessor = Preprocessor(
    crop=NoCrop(),
    mask=TomesLeeMask(),
    normalize=HuangNormalization(),
    filter=NoFilter(),
    )


from bob.bio.vein.extractor import RepeatedLineTracking
extractor = RepeatedLineTracking()

######################################################################    
all_feats=[]
all_imgs_aug = np.load('all_imgs_aug.npy') 

print("preparing data")
mylog_csv_path="my_log_RLT"
mylog=open(mylog_csv_path,'w')
mylog.close()

for i in range(all_imgs_aug.shape[0]):
    image = all_imgs_aug[i,:,:,0]*255.0
    image_and_mask = preprocessor(image)
    feature = extractor(image_and_mask)
    
    all_feats.append(feature)
    
    mylog=open(mylog_csv_path,'a')
    print('%d/%d' % (i+1, all_imgs_aug.shape[0]), end='\r',  file=mylog)
    mylog.close()
       
all_feats = np.array(all_feats)

np.save('all_rlt_aug_feats.npy', all_feats) 

######################################################################

######### Keras
#all_imgs  = np.reshape(all_imgs, (all_imgs.shape[0], all_imgs.shape[1], all_imgs.shape[2],1))
#all_feats = np.reshape(all_feats,(all_feats.shape[0], all_feats.shape[1], all_feats.shape[2],1))
######### PyTorch
#all_imgs  = np.reshape(all_imgs, (all_imgs.shape[0], 1, all_imgs.shape[1], all_imgs.shape[2]))
#all_feats = np.reshape(all_feats,(all_feats.shape[0], 1, all_feats.shape[1], all_feats.shape[2]))
