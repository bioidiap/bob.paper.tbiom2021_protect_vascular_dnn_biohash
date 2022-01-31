""" Configuration for BioHashing fingervein features extracted using the Wide Line Detector feature extractor, for the UTFVP database """ 

# TO DO BY USER: Define BioHashing parameters:
# **************************************************************************************
# Please modify the SCENARIO and LENGTH parameters according to your requirements: 
SCENARIO = 'st'  # 'n' for Normal, or 'st' for Stolen Token
LENGTH = 50  # BioHash length (i.e., number of bits in the resulting BioHash vector)
# **************************************************************************************


# Database: 
from bob.bio.vein.configurations.utfvp import database
protocol = 'nom'

# Directory where results will be placed:
temp_directory = './results'    
sub_directory = 'wld-AE-BHsh-50-stolen/baseline'  # pre-processed and extracted features will be placed here, along with the enrolled models
result_directory = temp_directory  # Miura match scores will be placed here  
from bob.io.base import create_directories_safe
create_directories_safe(temp_directory)

# Pre-processing based on locating the finger in the image and horizontally aligning it:
from bob.bio.vein.configurations.wide_line_detector import preprocessor



# Feature extraction based on the Wide Line Detector algorithm:
#from bob.bio.vein.configurations.wide_line_detector import extractor

#########################################load model###############################
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Conv2d(16,32,3,2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Conv2d(32,64,3,2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Conv2d(64,128,3,2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Conv2d(128,256,3,2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(3*5*256,500),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(0.2)
        )
        
        self.fc = nn.Sequential( 
            nn.Linear(500,3*5*256),
            #nn.BatchNorm2d(6*10*128),
            nn.ReLU(),
            #nn.Dropout(0.2)
        )
        
        self.decoder = nn.Sequential( 
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()
        )


    def forward(self, anchor, positive, negative):
        embedded_anchor   = self.encoder(anchor)
        embedded_positive = self.encoder(positive)
        embedded_negative = self.encoder(negative)
        
        input_to_decoder_anchor   = self.fc(embedded_anchor).view(-1,256,3,5)
        input_to_decoder_positive = self.fc(embedded_anchor).view(-1,256,3,5)
        input_to_decoder_negative = self.fc(embedded_anchor).view(-1,256,3,5)
        
        decoded_anchor   = self.decoder(input_to_decoder_anchor)
        decoded_positive = self.decoder(input_to_decoder_positive)
        decoded_negative = self.decoder(input_to_decoder_negative)
        
        return decoded_anchor, decoded_positive, decoded_negative, embedded_anchor, embedded_positive, embedded_negative
        
    def get_embeding(self,x):
        return self.encoder(x)
        
model = Net()
model = model.to(device)
epoch=90
model.load_state_dict(torch.load( '../model/{}.pth'.format(epoch)))
model.eval()
######################################

import numpy
import scipy
import scipy.misc

import bob.io.base
import bob.ip.base

from bob.bio.base.extractor import Extractor
import numpy as np

class WideLineDetector (Extractor):
    """Wide Line Detector feature extractor

    Based on B. Huang, Y. Dai, R. Li, D. Tang and W. Li. Finger-vein
    authentication based on wide line detector and pattern normalization,
    Proceedings on 20th International Conference on Pattern Recognition (ICPR),
    2010.
    """

    def __init__(
        self,
        radius = 5,    #Radius of the circular neighbourhood region
        threshold = 1, #Neigborhood threshold
        g = 41,         #Sum of neigbourhood threshold
        rescale = True,
        ):

        # call base class constructor
        Extractor.__init__(
        self,
        radius = radius,
        threshold = threshold,
        g = g,
        rescale = rescale,
        )

        # block parameters
        self.radius = radius
        self.threshold = threshold
        self.g = g
        self.rescale = rescale


    def wide_line_detector(self, finger_image, mask):
        """
        Computes and returns the Wide Line Detector features for the given input
        fingervein image
        """

        finger_image = finger_image.astype(numpy.float64)

        finger_mask = numpy.zeros(mask.shape)
        finger_mask[mask == True] = 1

        # Rescale image if required
        if self.rescale == True:
            scaling_factor = 0.24
            #finger_image = scipy.misc.imresize(finger_image,scaling_factor).astype()
            finger_image = bob.ip.base.scale(finger_image,scaling_factor)
            #finger_mask = scipy.misc.imresize(finger_mask,scaling_factor)
            finger_mask = bob.ip.base.scale(finger_mask,scaling_factor)
            #To eliminate residuals from the scalation of the binary mask
            finger_mask = scipy.ndimage.binary_dilation(finger_mask, structure=numpy.ones((1,1))).astype(int)

        x = numpy.arange((-1)*self.radius, self.radius+1)
        y = numpy.arange((-1)*self.radius, self.radius+1)
        X, Y = numpy.meshgrid(x,y)

        N = X**2 + Y**2 <= self.radius**2  # Neighbourhood mask

        img_h, img_w = finger_image.shape  #Image height and width

        veins = numpy.zeros(finger_image.shape)

        for y in range(self.radius,img_h-self.radius):
            for x in range(self.radius,img_w-self.radius):
                s=((finger_image[y-self.radius:y+self.radius+1,x-self.radius:x+self.radius+1] - finger_image[y,x]) <= self.threshold)
                m = (s*N).sum()
                veins[y,x] = float(m <= self.g)

        # Mask the vein image with the finger region
        img_veins_bin = veins*finger_mask
        
        img_veins_bin_=np.zeros([1,1,img_veins_bin.shape[0]+2,img_veins_bin.shape[1]-4])
        img_veins_bin_[0,0,1:-1,:]=img_veins_bin[:,2:-2]#/255.
        img_veins_bin_ = torch.tensor(img_veins_bin_, requires_grad=False).float().cuda()
        
        retval = model.get_embeding(img_veins_bin_).cpu().detach().numpy()
        
        return retval

    
    def __call__(self, image):
        """Reads the input image, extract the features based on Wide Line Detector
        of the fingervein image, and writes the resulting template"""
        #For debugging

        finger_image = image[0]    #Normalized image with histogram equalization
        finger_mask = image[1]

        return self.wide_line_detector(finger_image, finger_mask)

extractor=WideLineDetector()


# Algorithm:
# Miura matching:
#from bob.bio.vein.configurations.wide_line_detector import algorithm

import numpy
import scipy.spatial

from bob.bio.base.tools.FileSelector import FileSelector 
from bob.bio.base.algorithm import Algorithm
from bob.bio.base.extractor import Linearize


class BioHash (Algorithm):
    """This class defines a biometric template protection scheme based on BioHashing.

    **Parameters:**

    ``performs_projection`` : bool
    Set this flag to ``True`` to indicate that the BioHashing is performed in the projection stage

    ``requires_projector_training`` : bool
    Set this flag to ``False`` since we do not need to train the projector

    ``user_seed`` : int
    Integer specifying the seed to use to generate the BioHash projection matrix for each user in the Stolen Token scenario

    ``bh_length`` : int
    Integer specifying the desired length (number of bits) of the generated BioHash vector

    ``kwargs`` : ``key=value`` pairs
    A list of keyword arguments directly passed to the :py:class:`Algorithm` base class constructor.
    """

    def __init__(
      self,
      performs_projection=True,  # 'projection' is where BioHashing is performed
      requires_projector_training=False,
      user_seed=None,  # value passed for Stolen Token scenario only
      bh_length=None,
      **kwargs  # parameters directly sent to the base class
    ):

    # call base class constructor and register that the algorithm performs a projection
        super(BioHash, self).__init__(
            performs_projection = performs_projection,  # 'projection' is where BioHashing is performed
            requires_projector_training = requires_projector_training,
            **kwargs
        )

        self.user_seed = user_seed
        self.bh_length = bh_length
        if self.user_seed == None:  # Normal scenario
            # Initialize the file selector only so we can get client ids
            from bob.bio.vein.configurations.utfvp import database
            database.protocol = 'nom'
            FileSelector.create(
              database=database,
              extractor_file='',
              projector_file='',
              enroller_file='',
              preprocessed_directory='',
              extracted_directory='',
              projected_directory='',
              model_directories='',
              score_directories='',
              zt_score_directories='',
              compressed_extension='',
              default_extension='',
              zt_norm = None
            )
            fs = FileSelector.instance()
            self.original_data_files = fs.original_data_list(groups=['dev','eval']) 
            self.sample_id = -1  # initialise to -1 so that the first time sample_id is incremented it will be 0


    def create_biohash(self, feat_vec, bh_len, user_seed):
        """ Creates a BioHash by projecting the input biometric feature vector onto a set of randomly generated basis vectors and then binarising the resulting vector

        **Parameters:**

        feat_vec (array): The extracted fingervein feture vector

        bh_len (int): The desired length (i.e., number of bits) of the resulting BioHash

        user_seed (int): The seed used to generate the user's specific random projection matrix

        **Returns:**

        biohash (array): The resulting BioHash, which is a protected, binary representation of the input feature vector

        """

        numpy.random.seed(user_seed) # re-seed the random number generator according to the user's specific seed
        rand_mat = numpy.random.rand(len(feat_vec), bh_len) # generate matrix of random values from uniform distribution over [0, 1] 
        orth_mat, _ = numpy.linalg.qr(rand_mat, mode='reduced') # orthonormalise columns of random matrix, mode='reduced' returns orth_mat with size len(feat_vec) x bh_len    
        #biohash = [None]*bh_len  # initialize BioHash vector with empty elements
        #for c in range(bh_len):
        #    biohash[c] = numpy.dot(feat_vec, orth_mat[:, c]) # dot product between feature vector and each column of orthonormalised random matrix
        biohash = numpy.dot(feat_vec, orth_mat)
        thresh = numpy.mean(biohash) # threshold by which to binarise vector of dot products to generate final BioHash
        biohash = numpy.where(biohash > thresh, 1, 0)
        return biohash


    def project(self, feature):
        """project(feature) -> projected

        This function will project the given feature.  In this case, projection involves BioHashing the extracted fingervein images.

        **Parameters:**

        feature : object
          The feature to be projected (BioHashed).

        client_id : string
          The ID of the biometric sample whose feature we are projecting (BioHashing).

        **Returns:**

        projected : object
          The BioHashed features.

        """

        # BioHashing
        linearize_extractor = Linearize()
        feat_vec = linearize_extractor(feature)
        if self.user_seed == None: # normal scenario, so user_seed = client_id
            self.sample_id = self.sample_id + 1  # increment each time "project" method is called
            print("Current sample id = %s" % (self.sample_id))
            file_object = self.original_data_files[self.sample_id]
            user_seed = int(''.join([str(ch-48) for ch in (''.join(file_object.client_id.split('_'))).encode('ascii')])) # e.g., client_id 25_1 becomes user_seed 251, client_id 001_L -> 001_28 -> user_seed 128       
            print("client_id = %s" % (file_object.client_id))
            print("NORMAL scenario user seed: %s\n" % (user_seed))
            bh = self.create_biohash(feat_vec, self.bh_length, user_seed)
        else: # stolen token scenario, so user_seed will be some randomly generated number (same for every person in the database), specified in config file
            print("STOLEN TOKEN scenario user seed: %s\n" % (self.user_seed))
            bh = self.create_biohash(feat_vec, self.bh_length, self.user_seed)
        return bh


    def enroll(self, enroll_features):
        """enroll(enroll_features) -> model

        Enrolls the model BioHash by storing all given input vectors.

        **Parameters:**

        ``enroll_features`` : [:py:class:`numpy.ndarray`]
          The list of BioHashes to enroll the model from.

        **Returns:**

        ``model`` : 2D :py:class:`numpy.ndarray`
          The enrolled BioHash model.
        """
        return numpy.vstack(f.flatten() for f in enroll_features)


    def score(self, model, probe):
        """score(model, probe) -> score

        This function will compute the Hamming distance between the given model and probe BioHashes.

        **Parameters:**

        model : object
        The model BioHash to compare the probe BioHash with.
        The ``model`` was read using the :py:meth:`read_model` function.

        probe : object
        The probe BioHash to compare the model BioHash with.
        The ``probe`` was read using the :py:meth:`read_feature` function

        **Returns:**

        score : float
        The Hamming distance between ``model`` and ``probe``.
        Higher values define higher similarities.
        """

        probe = probe.flatten()
        if model.ndim == 2:
            # we have multiple models, so we use the multiple model scoring
            return self.score_for_multiple_models(model, probe)
        else:
            return scipy.spatial.distance.hamming(model, probe) * -1


if SCENARIO == 'n':
    USER_SEED = None  # the user seed in the Normal scenario is simply the client id
elif SCENARIO == 'st':
    USER_SEED = 100  # the user seed in the Stolen Token scenario is the same for every biometric sample 
algorithm = BioHash(bh_length=LENGTH, user_seed=USER_SEED)


# Set up parallel processing using all available processors:
#from bob.bio.vein.configurations.parallel import parallel, nice

# Specify the level of detail to be output in the terminal during program execution:
verbose = 2

# Specify which group of fingers to use:
groups = ['dev','eval']