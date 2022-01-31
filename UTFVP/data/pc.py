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


# from bob.bio.vein.extractor import MaximumCurvature
# extractor = MaximumCurvature()

##########
import numpy

import bob.io.base

from bob.bio.base.extractor import Extractor

from scipy.ndimage import gaussian_filter


class PrincipalCurvature (Extractor):
    """MiuraMax feature extractor

    Based on J.H. Choi, W. Song, T. Kim, S.R. Lee and H.C. Kim, Finger vein
    extraction using gradient normalization and principal curvature. Proceedings
    on Image Processing: Machine Vision Applications II, SPIE 7251, (2009)
    """

    def __init__(
          self,
          sigma = 3, # Gaussian standard deviation applied
          threshold = 2, # Percentage of maximum used for hard thresholding
          ):

        # call base class constructor
        Extractor.__init__(
            self,
            sigma = sigma,
            threshold = threshold,
            )

        # block parameters
        self.sigma = sigma
        self.threshold = threshold


    def ut_gauss(self,img,sigma,dx,dy):
        return gaussian_filter(numpy.float64(img), sigma, order = [dx,dy])

    def principal_curvature(self, image, mask):
        """Computes and returns the Maximum Curvature features for the given input
        fingervein image"""

        finger_mask = numpy.zeros(mask.shape)
        finger_mask[mask == True] = 1

        sigma = numpy.sqrt(self.sigma**2/2)

        gx = self.ut_gauss(image,self.sigma,1,0)
        gy = self.ut_gauss(image,self.sigma,0,1)

        Gmag = numpy.sqrt(gx**2 + gy**2) #  Gradient magnitude

        # Apply threshold
        gamma = (self.threshold/100)*numpy.max(Gmag)

        indices = numpy.where(Gmag < gamma)

        gx[indices] = 0
        gy[indices] = 0

        # Normalize
        Gmag[numpy.where(Gmag==0)] = 1  # Avoid dividing by zero
        gx = gx/Gmag
        gy = gy/Gmag

        hxx = self.ut_gauss(gx,sigma,1,0)
        hxy = self.ut_gauss(gx,sigma,0,1)
        hyy = self.ut_gauss(gy,sigma,0,1)

        lambda1 = 0.5*(hxx + hyy + numpy.sqrt(hxx**2 + hyy**2 - 2*hxx*hyy + 4*hxy**2))
        veins = lambda1*finger_mask

        # Normalise
        veins = veins - numpy.min(veins[:])
        veins = veins/numpy.max(veins[:])

        veins = veins*finger_mask



        # Binarise the vein image by otsu
        md = numpy.median(veins[veins>0])
        img_veins_bin = veins > md

        return img_veins_bin.astype(numpy.float64)


    def __call__(self, image):
        """Reads the input image, extract the features based on Principal Curvature
        of the fingervein image, and writes the resulting template"""

        finger_image = image[0]    #Normalized image with or without histogram equalization
        finger_mask = image[1]

        return self.principal_curvature(finger_image, finger_mask)

extractor = PrincipalCurvature(sigma = 6, threshold = 4)
##########

######################################################################    
all_feats=[]
all_imgs_aug = np.load('all_imgs_aug.npy') 

print("preparing data")
mylog_csv_path="my_log_pc"
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

np.save('all_pc_aug_feats.npy', all_feats) 

######################################################################

######### Keras
#all_imgs  = np.reshape(all_imgs, (all_imgs.shape[0], all_imgs.shape[1], all_imgs.shape[2],1))
#all_feats = np.reshape(all_feats,(all_feats.shape[0], all_feats.shape[1], all_feats.shape[2],1))
######### PyTorch
#all_imgs  = np.reshape(all_imgs, (all_imgs.shape[0], 1, all_imgs.shape[1], all_imgs.shape[2]))
#all_feats = np.reshape(all_feats,(all_feats.shape[0], 1, all_feats.shape[1], all_feats.shape[2]))