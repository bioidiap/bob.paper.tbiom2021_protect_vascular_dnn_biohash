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


from bob.bio.vein.extractor import WideLineDetector
extractor = WideLineDetector()

######################################################################    
all_feats=[]
all_imgs_aug = np.load('all_imgs_aug.npy') 

print("preparing data")
mylog_csv_path="my_log_wld"
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

np.save('all_wld_aug_feats.npy', all_feats) 