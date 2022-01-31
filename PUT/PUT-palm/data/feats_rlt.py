import numpy as np
from tqdm import tqdm
from bob.bio.vein.preprocessor import NoCrop, TomesLeeMask, HuangNormalization, \
    NoFilter, Preprocessor

import cv2
class resize_and_gray():
    def __call__(self,image):
        #image= np.transpose(image,(1,2,0))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(image, (0,0), fx=0.5, fy=0.5) 

preprocessor = Preprocessor(
    crop=resize_and_gray(),#NoCrop(),
    mask=TomesLeeMask(),
    normalize=HuangNormalization(),
    filter=NoFilter(),
    )

######################################################################    

from bob.bio.vein.extractor import WideLineDetector
wld_extractor = WideLineDetector()

from bob.bio.vein.extractor import MaximumCurvature
mc_extractor = MaximumCurvature()

from bob.bio.vein.extractor import RepeatedLineTracking
rlt_extractor = RepeatedLineTracking()
######################################################################    
# all_feats_wld=[]
# all_feats_mc =[]
all_feats_rlt=[]
# all_imgs_prep=[]
all_imgs_aug = np.load('palm_all_imgs_aug.npy') 

print("preparing data")
mylog_csv_path="my_log_rlt"
mylog=open(mylog_csv_path,'w')
mylog.close()

for i in tqdm(range(all_imgs_aug.shape[0])):
    image = all_imgs_aug[i,:,:,0]*255.0
    image_and_mask = preprocessor(image)
    
#     wld = wld_extractor(image_and_mask)
#     mc  = mc_extractor (image_and_mask)
    rlt = rlt_extractor(image_and_mask)
    
#     all_feats_wld.append(wld)
#     all_feats_mc.append(mc)
    all_feats_rlt.append(rlt)
#     all_imgs_prep.append(image_and_mask[0])
    
    mylog=open(mylog_csv_path,'a')
    print('%d/%d' % (i+1, all_imgs_aug.shape[0]), end='\r',  file=mylog)
    mylog.close()
       
# all_feats_wld=np.array(all_feats_wld)
# all_feats_mc=np.array (all_feats_mc)
all_feats_rlt=np.array(all_feats_rlt)
# all_imgs_prep=np.array(all_imgs_prep)

# np.save('palm_all_wld_aug_feats.npy', all_feats_wld) 
# np.save('palm_all_mc_aug_feats.npy', all_feats_mc) 
np.save('palm_all_rlt_aug_feats.npy', all_feats_rlt) 
# np.save('palm_all_imgs_aug_prep.npy', all_imgs_prep) 

######################################################################