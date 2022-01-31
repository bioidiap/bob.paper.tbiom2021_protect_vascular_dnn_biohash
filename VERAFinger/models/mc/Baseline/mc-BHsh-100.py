""" Configuration for extracting fingervein features using the MC feature extractor and calculating the baseline match scores, for the VERA Finger Vein database """ 

# TO DO BY USER: Define BioHashing parameters:
# **************************************************************************************
# Please modify the SCENARIO and LENGTH parameters according to your requirements: 
SCENARIO = 'n'  # 'n' for Normal, or 'st' for Stolen Token
LENGTH = 100  # BioHash length (i.e., number of bits in the resulting BioHash vector)
# **************************************************************************************



# Database: 
#from bob.bio.vein.configurations.verafinger import database
"""
I used the source code to remove presentation attack data from dataset.
"""

import os
import numpy
class AnnotatedArray(numpy.ndarray):
    """Defines a numpy array subclass that can carry its own metadata

  Copied from: https://docs.scipy.org/doc/numpy-1.12.0/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
  """

    def __new__(cls, input_array, metadata=None):
        obj = numpy.asarray(input_array).view(cls)
        obj.metadata = metadata if metadata is not None else dict()
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', dict())

from bob.bio.base.database import BioFile, BioDatabase

class File(BioFile):
    """
    Implements extra properties of vein files for the Vera Fingervein database


    Parameters:

        f (object): Low-level file (or sample) object that is kept inside

    """

    def __init__(self, f):

        id_ = f.finger.unique_name
        if f.source == 'pa': id_ = 'attack/%s' % id_
        super(File, self).__init__(client_id=id_, path=f.path, file_id=f.id)
        self.__f = f


    def load(self, *args, **kwargs):
        """(Overrides base method) Loads both image and mask"""

        image = super(File, self).load(*args, **kwargs)
        basedir = args[0] if args else kwargs['directory']
        annotdir = os.path.join(basedir, 'annotations', 'roi')
        if os.path.exists(annotdir):
            roi = self.__f.roi(args[0])
            return AnnotatedArray(image, metadata=dict(roi=roi))
        return image


class Database(BioDatabase):
    """
    Implements verification API for querying Vera Fingervein database.
    """

    def __init__(self, **kwargs):

        super(Database, self).__init__(name='verafinger', **kwargs)
        from bob.db.verafinger.query import Database as LowLevelDatabase
        self._db = LowLevelDatabase()

        self.low_level_group_names = ('train', 'dev')
        self.high_level_group_names = ('world', 'dev')

    def groups(self):

        return self.convert_names_to_highlevel(self._db.groups(),
            self.low_level_group_names, self.high_level_group_names)

    def client_id_from_model_id(self, model_id, group='dev'):
        """Required as ``model_id != client_id`` on this database"""

        return self._db.finger_name_from_model_id(model_id)


    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):

        groups = self.convert_names_to_lowlevel(groups,
            self.low_level_group_names, self.high_level_group_names)
        if protocol.endswith('-va') or protocol.endswith('-VA'):
            protocol = protocol[:-3]
        return self._db.model_ids(groups=groups, protocol=protocol)


    def objects(self, groups=None, protocol=None, purposes=None,
                model_ids=None, **kwargs):

        groups = self.convert_names_to_lowlevel(groups,
            self.low_level_group_names, self.high_level_group_names)

        if protocol.endswith('-va') or protocol.endswith('-VA'):
            protocol = protocol[:-3]
            if purposes=='probe': purposes='attack'

        retval = self._db.objects(groups=groups, protocol=protocol,
            purposes=purposes, model_ids=model_ids, **kwargs)

        return [File(f) for f in retval if f.source == 'bf']


    def annotations(self, file):
        return None

_verafinger_directory = "[YOUR_VERAFINGER_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

database = Database(
    original_directory = _verafinger_directory,
    original_extension = '.png',
    )
protocol = 'Fifty'

# Directory where results will be placed:
temp_directory = './results'   
sub_directory = 'mc-BHsh-100/baseline'  # pre-processed and extracted features will be placed here, along with the enrolled models
result_directory = temp_directory  # Miura match scores will be placed here  
from bob.io.base import create_directories_safe
create_directories_safe(temp_directory)



# Pre-processing based on locating the finger in the image and horizontally aligning it:
from bob.bio.vein.preprocessor import NoCrop, TomesLeeMask, HuangNormalization, \
    NoFilter, Preprocessor

preprocessor = Preprocessor(
    crop=NoCrop(),
    mask=TomesLeeMask(),
    normalize=HuangNormalization(),
    filter=NoFilter(),
    )


# Feature extraction based on the Maximum Curvature algorithm:
from bob.bio.vein.configurations.maximum_curvature import extractor

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
            #from bob.bio.vein.configurations.verafinger import database
            database = Database(
                            original_directory = _verafinger_directory,
                            original_extension = '.png',
                            )
            database.protocol = 'Fifty'
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
            self.original_data_files = fs.original_data_list(groups=['dev']) 
#             self.original_data_files = []
#             for file in original_data_files:
#                 if file.path[5:7]=='bf':
#                     self.original_data_files.append(file)
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
            print("client_id = %s" % (file_object.client_id))
            #user_seed = file_object.client_id   
            user_seed = int(''.join([str(ch-48) for ch in (''.join(file_object.client_id.split('_'))).encode('ascii')])) # e.g.,  client_id 001_L -> 001_28 -> user_seed 128
            #print(file_object.client_id,user_seed, len(self.original_data_files))
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
groups = ['dev']