""" Configuration for extracting fingervein features using the Wide Line Detector feature extractor and calculating the baseline match scores, for the PUTVein database """ 

# TO DO BY USER: Define BioHashing parameters:
# **************************************************************************************
# Please modify the SCENARIO and LENGTH parameters according to your requirements: 
SCENARIO = 'n'  # 'n' for Normal, or 'st' for Stolen Token
LENGTH = 100  # BioHash length (i.e., number of bits in the resulting BioHash vector)
# **************************************************************************************



# Database: 
from bob.bio.vein.configurations.putvein import database
protocol = 'palm-R_1'

# Directory where results will be placed:
temp_directory = './results'   
sub_directory = 'wld-BHsh-100/baseline'  # pre-processed and extracted features will be placed here, along with the enrolled models
result_directory = temp_directory  # Miura match scores will be placed here  
from bob.io.base import create_directories_safe
create_directories_safe(temp_directory)



# Pre-processing based on locating the finger in the image and horizontally aligning it:
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


# Feature extraction based on the Wide Line Detector algorithm:
from bob.bio.vein.configurations.wide_line_detector import extractor


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
            from bob.bio.vein.configurations.putvein import database
            database.protocol = 'palm-R_1'
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
            user_seed = file_object.client_id   
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