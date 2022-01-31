""" Configuration for BioHashing fingervein features extracted using the Repeated Line Tracking feature extractor, for the UTFVP database """ 

# TO DO BY USER: Define BioHashing parameters:
# **************************************************************************************
# Please modify the SCENARIO and LENGTH parameters according to your requirements: 
SCENARIO = 'st'  # 'n' for Normal, or 'st' for Stolen Token
LENGTH = 100  # BioHash length (i.e., number of bits in the resulting BioHash vector)
# **************************************************************************************


# Database: 
from bob.bio.vein.configurations.utfvp import database
protocol = 'nom'

# Directory where results will be placed:
temp_directory = './results'    
sub_directory = 'rlt-PCA-BHsh-100-stolen/baseline'  # pre-processed and extracted features will be placed here, along with the enrolled models
result_directory = temp_directory  # Miura match scores will be placed here  
from bob.io.base import create_directories_safe
create_directories_safe(temp_directory)

# Pre-processing based on locating the finger in the image and horizontally aligning it:
from bob.bio.vein.configurations.maximum_curvature import preprocessor

# Algorithm:
# Miura matching:
#from bob.bio.vein.configurations.maximum_curvature import algorithm



import pickle
with open('pca.pkl','rb') as f:
    pca_model = pickle.load(f)


import numpy
import scipy
import scipy.misc

import bob.io.base
import bob.ip.base

from bob.bio.base.extractor import Extractor
import numpy as np
import math


class RepeatedLineTracking (Extractor):
    """Repeated Line Tracking feature extractor

    Based on N. Miura, A. Nagasaka, and T. Miyatake. Feature extraction of finger
    vein patterns based on repeated line tracking and its application to personal
    identification. Machine Vision and Applications, Vol. 15, Num. 4, pp.
    194--203, 2004
    """

    def __init__(
                self,
                iterations = 3000, # Maximum number of iterations
                r = 1,             # Distance between tracking point and cross section of the profile
                profile_w = 21,    # Width of profile (Error: profile_w must be odd)
                rescale = True,
                seed = 0,          # Seed for the algorithm's random walk
                ):

        # call base class constructor
        Extractor.__init__(
            self,
            iterations = iterations,
            r = r,
            profile_w = profile_w,
            rescale = rescale,
            seed = seed,
            )

        # block parameters
        self.iterations = iterations
        self.r = r
        self.profile_w = profile_w
        self.rescale = rescale
        self.seed = seed


    def repeated_line_tracking(self, finger_image, mask):
        """Computes and returns the MiuraMax features for the given input
        fingervein image"""

        # Sets the random seed before starting to process
        numpy.random.seed(self.seed)

        finger_mask = numpy.zeros(mask.shape)
        finger_mask[mask == True] = 1

        # Rescale image if required
        if self.rescale == True:
            scaling_factor = 0.6
            finger_image = bob.ip.base.scale(finger_image,scaling_factor)
            finger_mask = bob.ip.base.scale(finger_mask,scaling_factor)
            #To eliminate residuals from the scalation of the binary mask
            finger_mask = scipy.ndimage.binary_dilation(finger_mask, structure=numpy.ones((1,1))).astype(int)

        p_lr = 0.5  # Probability of goin left or right
        p_ud = 0.25 # Probability of going up or down

        Tr = numpy.zeros(finger_image.shape) # Locus space
        filtermask = numpy.array(([-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]))

        # Check if progile w is even
        if (self.profile_w.__mod__(2) == 0):
            print ('Error: profile_w must be odd')

        ro = numpy.round(self.r*math.sqrt(2)/2)    # r for oblique directions
        hW = (self.profile_w-1)/2                  # half width for horz. and vert. directions
        hWo = numpy.round(hW*math.sqrt(2)/2)       # half width for oblique directions

        # Omit unreachable borders
        border = int(self.r+hW)
        finger_mask[0:border,:] = 0
        finger_mask[finger_mask.shape[0]-border:,:] = 0
        finger_mask[:,0:border] = 0
        finger_mask[:,finger_mask.shape[1]-border:] = 0

        ## Uniformly distributed starting points
        aux = numpy.argwhere( (finger_mask > 0) == True )
        indices = numpy.random.permutation(aux)
        indices = indices[0:self.iterations,:]    # Limit to number of iterations

        ## Iterate through all starting points
        for it in range(0,self.iterations):
            yc = indices[it,0] # Current tracking point, y
            xc = indices[it,1] # Current tracking point, x

            # Determine the moving-direction attributes
            # Going left or right ?
            if (numpy.random.random_sample() >= 0.5):
                Dlr = -1  # Going left
            else:
                Dlr = 1   # Going right

            # Going up or down ?
            if (numpy.random.random_sample() >= 0.5):
                Dud = -1  # Going up
            else:
                Dud = 1   # Going down

            # Initialize locus-positition table Tc
            Tc = numpy.zeros(finger_image.shape, numpy.bool)

            #Dlr = -1; Dud=-1; LET OP
            Vl = 1
            while (Vl > 0):
                # Determine the moving candidate point set Nc
                Nr = numpy.zeros([3,3], numpy.bool)
                Rnd = numpy.random.random_sample()
                #Rnd = 0.8 LET OP
                if (Rnd < p_lr):
                    # Going left or right
                    Nr[:,1+Dlr] = True
                elif (Rnd >= p_lr) and (Rnd < (p_lr + p_ud)):
                    # Going up or down
                    Nr[1+Dud,:] = True
                else:
                    # Going any direction
                    Nr = numpy.ones([3,3], numpy.bool)
                    Nr[1,1] = False
                #tmp = numpy.argwhere( (~Tc[yc-2:yc+1,xc-2:xc+1] & Nr & finger_mask[yc-2:yc+1,xc-2:xc+1].astype(numpy.bool)).T.reshape(-1) == True )
                tmp = numpy.argwhere( (~Tc[yc-1:yc+2,xc-1:xc+2] & Nr & finger_mask[yc-1:yc+2,xc-1:xc+2].astype(numpy.bool)).T.reshape(-1) == True )
                Nc = numpy.concatenate((xc + filtermask[tmp,0],yc + filtermask[tmp,1]),axis=1)
                if (Nc.size==0):
                    Vl=-1
                    continue

                ## Detect dark line direction near current tracking point
                Vdepths = numpy.zeros((Nc.shape[0],1)) # Valley depths
                for i in range(0,Nc.shape[0]):
                    ## Horizontal or vertical
                    if (Nc[i,1] == yc):
                        # Horizontal plane
                        yp = Nc[i,1]
                        if (Nc[i,0] > xc):
                            # Right direction
                            xp = Nc[i,0] + self.r
                        else:
                            # Left direction
                            xp = Nc[i,0] - self.r
                        Vdepths[i] = finger_image[int(yp + hW), int(xp)] - 2*finger_image[int(yp),int(xp)] + finger_image[int(yp - hW), int(xp)]
                    elif (Nc[i,0] == xc):
                        # Vertical plane
                        xp = Nc[i,0]
                        if (Nc[i,1] > yc):
                            # Down direction
                            yp = Nc[i,1] + self.r
                        else:
                            # Up direction
                            yp = Nc[i,1] - self.r
                        Vdepths[i] = finger_image[int(yp), int(xp + hW)] - 2*finger_image[int(yp),int(xp)] + finger_image[int(yp), int(xp - hW)]

                    ## Oblique directions
                    if ( (Nc[i,0] > xc) and (Nc[i,1] < yc) ) or ( (Nc[i,0] < xc) and (Nc[i,1] > yc) ):
                        # Diagonal, up /
                        if (Nc[i,0] > xc and Nc[i,1] < yc):
                            # Top right
                            xp = Nc[i,0] + ro
                            yp = Nc[i,1] - ro
                        else:
                            # Bottom left
                            xp = Nc[i,0] - ro
                            yp = Nc[i,1] + ro
                        Vdepths[i] = finger_image[int(yp - hWo), int(xp - hWo)] - 2*finger_image[int(yp),int(xp)] + finger_image[int(yp + hWo), int(xp + hWo)]
                    else:
                        # Diagonal, down \
                        if (Nc[i,0] < xc and Nc[i,1] < yc):
                            # Top left
                            xp = Nc[i,0] - ro
                            yp = Nc[i,1] - ro
                        else:
                            # Bottom right
                            xp = Nc[i,0] + ro
                            yp = Nc[i,1] + ro
                        Vdepths[i] = finger_image[int(yp + hWo), int(xp - hWo)] - 2*finger_image[int(yp),int(xp)] + finger_image[int(yp - hWo), int(xp + hWo)]
                # End search of candidates
                index = numpy.argmax(Vdepths)  #Determine best candidate
                # Register tracking information
                Tc[yc, xc] = True
                # Increase value of tracking space
                Tr[yc, xc] = Tr[yc, xc] + 1
                # Move tracking point
                xc = Nc[index, 0]
                yc = Nc[index, 1]

        img_veins = Tr

        # Binarise the vein image
        md = numpy.median(img_veins[img_veins>0])
        img_veins_bin = img_veins > md
        img_veins_bin = scipy.ndimage.binary_closing(img_veins_bin, structure=numpy.ones((2,2))).astype(int)

        return img_veins_bin.astype(numpy.float64)


    def skeletonize(self, img):
        import scipy.ndimage.morphology as m
        h1 = numpy.array([[0, 0, 0],[0, 1, 0],[1, 1, 1]])
        m1 = numpy.array([[1, 1, 1],[0, 0, 0],[0, 0, 0]])
        h2 = numpy.array([[0, 0, 0],[1, 1, 0],[0, 1, 0]])
        m2 = numpy.array([[0, 1, 1],[0, 0, 1],[0, 0, 0]])
        hit_list = []
        miss_list = []
        for k in range(4):
            hit_list.append(numpy.rot90(h1, k))
            hit_list.append(numpy.rot90(h2, k))
            miss_list.append(numpy.rot90(m1, k))
            miss_list.append(numpy.rot90(m2, k))
        img = img.copy()
        while True:
            last = img
            for hit, miss in zip(hit_list, miss_list):
                hm = m.binary_hit_or_miss(img, hit, miss)
                img = numpy.logical_and(img, numpy.logical_not(hm))
            if numpy.all(img == last):
                break
        return img


    def __call__(self, image):
        """Reads the input image, extract the features based on Maximum Curvature
        of the fingervein image, and writes the resulting template"""

        finger_image = image[0]    #Normalized image with or without histogram equalization
        finger_mask = image[1]

        rlt = self.repeated_line_tracking(finger_image, finger_mask)


        rlt=np.reshape(rlt,(1,-1))

        rlt_pca = pca_model.transform(rlt)

        return rlt_pca
    
extractor=RepeatedLineTracking()


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