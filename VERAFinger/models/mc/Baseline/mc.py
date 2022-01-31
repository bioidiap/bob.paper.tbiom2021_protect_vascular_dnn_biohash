""" Configuration for extracting fingervein features using the MC feature extractor and calculating the baseline match scores, for the VERA Finger Vein database """ 

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
sub_directory = 'mc/baseline'  # pre-processed and extracted features will be placed here, along with the enrolled models
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

import numpy
import scipy.signal

from bob.bio.base.algorithm import Algorithm


class MiuraMatch (Algorithm):
    """Finger vein matching: match ratio via cross-correlation

    The method is based on "cross-correlation" between a model and a probe image.
    It convolves the binary image(s) representing the model with the binary image
    representing the probe (rotated by 180 degrees), and evaluates how they
    cross-correlate. If the model and probe are very similar, the output of the
    correlation corresponds to a single scalar and approaches a maximum. The
    value is then normalized by the sum of the pixels lit in both binary images.
    Therefore, the output of this method is a floating-point number in the range
    :math:`[0, 0.5]`. The higher, the better match.

    In case model and probe represent images from the same vein structure, but
    are misaligned, the output is not guaranteed to be accurate. To mitigate this
    aspect, Miura et al. proposed to add a *small* cropping factor to the model
    image, assuming not much information is available on the borders (``ch``, for
    the vertical direction and ``cw``, for the horizontal direction). This allows
    the convolution to yield searches for different areas in the probe image. The
    maximum value is then taken from the resulting operation. The convolution
    result is normalized by the pixels lit in both the cropped model image and
    the matching pixels on the probe that yield the maximum on the resulting
    convolution.

    For this to work properly, input images are supposed to be binary in nature,
    with zeros and ones.

    Based on N. Miura, A. Nagasaka, and T. Miyatake. Feature extraction of finger
    vein patterns based on repeated line tracking and its application to personal
    identification. Machine Vision and Applications, Vol. 15, Num. 4, pp.
    194--203, 2004

    Parameters:

    ch (:py:class:`int`, optional): Maximum search displacement in y-direction.

    cw (:py:class:`int`, optional): Maximum search displacement in x-direction.

    """

    def __init__(self,
        ch = 80,       # Maximum search displacement in y-direction
        cw = 90,       # Maximum search displacement in x-direction
        ):

        # call base class constructor
        Algorithm.__init__(
            self,

            ch = ch,
            cw = cw,

            multiple_model_scoring = None,
            multiple_probe_scoring = None
            )

        self.ch = ch
        self.cw = cw


    def enroll(self, enroll_features):
        """Enrolls the model by computing an average graph for each model"""

        # return the generated model
        return numpy.array(enroll_features)


    def score(self, model, probe):
        """Computes the score between the probe and the model.

        Parameters:

          model (numpy.ndarray): The model of the user to test the probe agains

          probe (numpy.ndarray): The probe to test


        Returns:

          float: Value between 0 and 0.5, larger value means a better match

        """

        I=probe.astype(numpy.float64)

        if len(model.shape) == 2:
            model = numpy.array([model])

        scores = []

        # iterate over all models for a given individual
        for md in model:
            # erode model by (ch, cw)
            R = md.astype(numpy.float64)
            h, w = R.shape #same as I
            crop_R = R[self.ch:h-self.ch, self.cw:w-self.cw]

            # correlates using scipy - fastest option available iff the self.ch and
            # self.cw are height (>30). In this case, the number of components
            # returned by the convolution is high and using an FFT-based method
            # yields best results. Otherwise, you may try  the other options bellow
            # -> check our test_correlation() method on the test units for more
            # details and benchmarks.
            Nm = scipy.signal.fftconvolve(I, numpy.rot90(crop_R, k=2), 'valid')
            # 2nd best: use convolve2d or correlate2d directly;
            # Nm = scipy.signal.convolve2d(I, numpy.rot90(crop_R, k=2), 'valid')
            # 3rd best: use correlate2d
            # Nm = scipy.signal.correlate2d(I, crop_R, 'valid')

            # figures out where the maximum is on the resulting matrix
            t0, s0 = numpy.unravel_index(Nm.argmax(), Nm.shape)

            # this is our output
            Nmm = Nm[t0,s0]

            # normalizes the output by the number of pixels lit on the input
            # matrices, taking into consideration the surface that produced the
            # result (i.e., the eroded model and part of the probe)
            scores.append(Nmm/(crop_R.sum() + I[t0:t0+h-2*self.ch, s0:s0+w-2*self.cw].sum()))

        return numpy.mean(scores)

algorithm = MiuraMatch(ch=18, cw=28)


# Set up parallel processing using all available processors:
#from bob.bio.vein.configurations.parallel import parallel, nice

# Specify the level of detail to be output in the terminal during program execution:
verbose = 2


# Specify which group of fingers to use:
groups = ['dev']
