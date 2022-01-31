""" Configuration for BioHashing fingervein features extracted using the Wide Line Detector feature extractor, for the PUT database """ 

# TO DO BY USER: Define BioHashing parameters:
# **************************************************************************************
# Please modify the SCENARIO and LENGTH parameters according to your requirements: 
SCENARIO = 'n'  # 'n' for Normal, or 'st' for Stolen Token
LENGTH = 100  # BioHash length (i.e., number of bits in the resulting BioHash vector)
# **************************************************************************************


# Database: 
from bob.bio.vein.configurations.putvein import database
protocol = 'wrist-R_1'

# Directory where results will be placed:
temp_directory = './results'    
sub_directory = 'mc-AE-BHsh-100/baseline'  # pre-processed and extracted features will be placed here, along with the enrolled models
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
#from bob.bio.vein.configurations.wide_line_detector import extractor

#########################################load model###############################
import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)

embedding_layer_length=100

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
#             nn.Dropout(0.2),
            nn.Conv2d(16,32,3,2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
#             nn.Dropout(0.2),
            nn.Conv2d(32,64,3,2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
#             nn.Dropout(0.2),
            nn.Conv2d(64,128,3,2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
#             nn.Dropout(0.2),
            nn.Conv2d(128,256,3,2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
#             nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(12*16*256,embedding_layer_length),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(0.2)
        )
        
        self.fc = nn.Sequential( 
            nn.Linear(embedding_layer_length,12*16*256),
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
        
        input_to_decoder_anchor   = self.fc(embedded_anchor).view(-1,256,12,16)
        input_to_decoder_positive = self.fc(embedded_anchor).view(-1,256,12,16)
        input_to_decoder_negative = self.fc(embedded_anchor).view(-1,256,12,16)
        
        decoded_anchor   = self.decoder(input_to_decoder_anchor)
        decoded_positive = self.decoder(input_to_decoder_positive)
        decoded_negative = self.decoder(input_to_decoder_negative)
        
        return decoded_anchor, decoded_positive, decoded_negative, embedded_anchor, embedded_positive, embedded_negative
        
    def get_embeding(self,x):
        return self.encoder(x)
        
    
model = Net()
model.to(device)
epoch=98
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
import math



class MaximumCurvature (Extractor):
    """
    MiuraMax feature extractor.

    Based on N. Miura, A. Nagasaka, and T. Miyatake, Extraction of Finger-Vein
    Pattern Using Maximum Curvature Points in Image Profiles. Proceedings on IAPR
    conference on machine vision applications, 9 (2005), pp. 347--350.


    Parameters:

    sigma (:py:class:`int`, optional): standard deviation for the gaussian
      smoothing kernel used to denoise the input image. The width of the
      gaussian kernel will be set automatically to 4x this value (in pixels).

    """


    def __init__(self, sigma = 5):
        Extractor.__init__(self, sigma = sigma)
        self.sigma = sigma


    def detect_valleys(self, image, mask):
        """Detects valleys on the image respecting the mask

        This step corresponds to Step 1-1 in the original paper. The objective is,
        for all 4 cross-sections (z) of the image (horizontal, vertical, 45 and -45
        diagonals), to compute the following proposed valley detector as defined in
        Equation 1, page 348:

        .. math::

           \kappa(z) = \\frac{d^2P_f(z)/dz^2}{(1 + (dP_f(z)/dz)^2)^\\frac{3}{2}}


        We start the algorithm by smoothing the image with a 2-dimensional gaussian
        filter. The equation that defines the kernel for the filter is:

        .. math::

           \mathcal{N}(x,y)=\\frac{1}{2\pi\sigma^2}e^\\frac{-(x^2+y^2)}{2\sigma^2}


        This is done to avoid noise from the raw data (from the sensor). The
        maximum curvature method then requires we compute the first and second
        derivative of the image for all cross-sections, as per the equation above.

        We instead take the following equivalent approach:

        1. construct a gaussian filter
        2. take the first (dh/dx) and second (d^2/dh^2) deritivatives of the filter
        3. calculate the first and second derivatives of the smoothed signal using
           the results from 3. This is done for all directions we're interested in:
           horizontal, vertical and 2 diagonals. First and second derivatives of a
           convolved signal

        .. note::

           Item 3 above is only possible thanks to the steerable filter property of
           the gaussian kernel. See "The Design and Use of Steerable Filters" from
           Freeman and Adelson, IEEE Transactions on Pattern Analysis and Machine
           Intelligence, Vol. 13, No. 9, September 1991.


        Parameters:

          image (numpy.ndarray): an array of 64-bit floats containing the input
            image
          mask (numpy.ndarray): an array, of the same size as ``image``, containing
            a mask (booleans) indicating where the finger is on ``image``.


        Returns:

          numpy.ndarray: a 3-dimensional array of 64-bits containing $\kappa$ for
          all considered directions. $\kappa$ has the same shape as ``image``,
          except for the 3rd. dimension, which provides planes for the
          cross-section valley detections for each of the contemplated directions,
          in this order: horizontal, vertical, +45 degrees, -45 degrees.

        """

        # 1. constructs the 2D gaussian filter "h" given the window size,
        # extrapolated from the "sigma" parameter (4x)
        # N.B.: This is a text-book gaussian filter definition
        winsize = numpy.ceil(4*self.sigma) #enough space for the filter
        window = numpy.arange(-winsize, winsize+1)
        X, Y = numpy.meshgrid(window, window)
        G = 1.0 / (2*math.pi*self.sigma**2)
        G *= numpy.exp(-(X**2 + Y**2) / (2*self.sigma**2))

        # 2. calculates first and second derivatives of "G" with respect to "X"
        # (0), "Y" (90 degrees) and 45 degrees (?)
        G1_0 = (-X/(self.sigma**2))*G
        G2_0 = ((X**2 - self.sigma**2)/(self.sigma**4))*G
        G1_90 = G1_0.T
        G2_90 = G2_0.T
        hxy = ((X*Y)/(self.sigma**4))*G

        # 3. calculates derivatives w.r.t. to all directions of interest
        #    stores results in the variable "k". The entries (last dimension) in k
        #    correspond to curvature detectors in the following directions:
        #
        #    [0] horizontal
        #    [1] vertical
        #    [2] diagonal \ (45 degrees rotation)
        #    [3] diagonal / (-45 degrees rotation)
        image_g1_0  = scipy.ndimage.convolve(image, G1_0, mode='nearest')
        image_g2_0  = scipy.ndimage.convolve(image, G2_0, mode='nearest')
        image_g1_90 = scipy.ndimage.convolve(image, G1_90, mode='nearest')
        image_g2_90 = scipy.ndimage.convolve(image, G2_90, mode='nearest')
        fxy = scipy.ndimage.convolve(image, hxy, mode='nearest')

        # support calculation for diagonals, given the gaussian kernel is
        # steerable. To calculate the derivatives for the "\" diagonal, we first
        # **would** have to rotate the image 45 degrees counter-clockwise (so the
        # diagonal lies on the horizontal axis). Using the steerable property, we
        # can evaluate the first derivative like this:
        #
        # image_g1_45 = cos(45)*image_g1_0 + sin(45)*image_g1_90
        #             = sqrt(2)/2*fx + sqrt(2)/2*fx
        #
        # to calculate the first derivative for the "/" diagonal, we first
        # **would** have to rotate the image -45 degrees "counter"-clockwise.
        # Therefore, we can calculate it like this:
        #
        # image_g1_m45 = cos(-45)*image_g1_0 + sin(-45)*image_g1_90
        #              = sqrt(2)/2*image_g1_0 - sqrt(2)/2*image_g1_90
        #

        image_g1_45 = 0.5*numpy.sqrt(2)*(image_g1_0 + image_g1_90)
        image_g1_m45  = 0.5*numpy.sqrt(2)*(image_g1_0 - image_g1_90)

        # NOTE: You can't really get image_g2_45 and image_g2_m45 from the theory
        # of steerable filters. In contact with B.Ton, he suggested the following
        # material, where that is explained: Chapter 5.2.3 of van der Heijden, F.
        # (1994) Image based measurement systems: object recognition and parameter
        # estimation. John Wiley & Sons Ltd, Chichester. ISBN 978-0-471-95062-2

        # This also shows the same result:
        # http://www.mif.vu.lt/atpazinimas/dip/FIP/fip-Derivati.html (look for
        # SDGD)

        # He also suggested to look at slide 75 of the following presentation
        # indicating it is self-explanatory: http://slideplayer.com/slide/5084635/

        image_g2_45 = 0.5*image_g2_0 + fxy + 0.5*image_g2_90
        image_g2_m45  = 0.5*image_g2_0 - fxy + 0.5*image_g2_90

        # ######################################################################
        # [Step 1-1] Calculation of curvature profiles
        # ######################################################################

        # Peak detection (k or kappa) calculation as per equation (1) page 348 on
        # Miura's paper
        finger_mask = mask.astype('float64')

        return numpy.dstack([
          (image_g2_0   / ((1 + image_g1_0**2)**(1.5))  ) * finger_mask,
          (image_g2_90  / ((1 + image_g1_90**2)**(1.5)) ) * finger_mask,
          (image_g2_45  / ((1 + image_g1_45**2)**(1.5)) ) * finger_mask,
          (image_g2_m45 / ((1 + image_g1_m45**2)**(1.5))) * finger_mask,
          ])


    def eval_vein_probabilities(self, k):
        '''Evaluates joint vein centre probabilities from cross-sections

        This function will take $\kappa$ and will calculate the vein centre
        probabilities taking into consideration valley widths and depths. It
        aggregates the following steps from the paper:

        * [Step 1-2] Detection of the centres of veins
        * [Step 1-3] Assignment of scores to the centre positions
        * [Step 1-4] Calculation of all the profiles

        Once the arrays of curvatures (concavities) are calculated, here is how
        detection works: The code scans the image in a precise direction (vertical,
        horizontal, diagonal, etc). It tries to find a concavity on that direction
        and measure its width (see Wr on Figure 3 on the original paper). It then
        identifies the centers of the concavity and assign a value to it, which
        depends on its width (Wr) and maximum depth (where the peak of darkness
        occurs) in such a concavity. This value is accumulated on a variable (Vt),
        which is re-used for all directions. Vt represents the vein probabilites
        from the paper.


        Parameters:

          k (numpy.ndarray): a 3-dimensional array of 64-bits containing $\kappa$
            for all considered directions. $\kappa$ has the same shape as
            ``image``, except for the 3rd. dimension, which provides planes for the
            cross-section valley detections for each of the contemplated
            directions, in this order: horizontal, vertical, +45 degrees, -45
            degrees.


        Returns:

          numpy.ndarray: The un-accumulated vein centre probabilities ``V``. This
          is a 3D array with 64-bit floats with the same dimensions of the input
          array ``k``. You must accumulate (sum) over the last dimension to
          retrieve the variable ``V`` from the paper.

        '''

        V = numpy.zeros(k.shape[:2], dtype='float64')

        def _prob_1d(a):
            '''Finds "vein probabilities" in a 1-D signal

            This function efficiently counts the width and height of concavities in
            the cross-section (1-D) curvature signal ``s``.

            It works like this:

            1. We create a 1-shift difference between the thresholded signal and
             itself
            2. We compensate for starting and ending regions
            3. For each sequence of start/ends, we compute the maximum in the
             original signal

            Example (mixed with pseudo-code):

             a = 0 1 2 3 2 1 0 -1 0 0 1 2 5 2 2 2 1
             b = a > 0 (as type int)
             b = 0 1 1 1 1 1 0  0 0 0 1 1 1 1 1 1 1

             0 1 1 1 1 1  0 0 0 0 1 1 1 1 1 1 1
               0 1 1 1 1  1 0 0 0 0 1 1 1 1 1 1 1 (-)
            -------------------------------------------
             X 1 0 0 0 0 -1 0 0 0 1 0 0 0 0 0 0 X (length is smaller than orig.)

             starts = numpy.where(diff > 0)
             ends   = numpy.where(diff < 0)

             -> now the number of starts and ends should match, otherwise, we must
             compensate

                -> case 1: b starts with 1: add one start in begin of "starts"
                -> case 2: b ends with 1: add one end in the end of "ends"

             -> iterate over the sequence of starts/ends and find maximums


            Parameters:

            a (numpy.ndarray): 1D signal with curvature to explore


            Returns:

            numpy.ndarray: 1D container with the vein centre probabilities

            '''

            b = (a > 0).astype(int)
            diff = b[1:] - b[:-1]
            starts = numpy.argwhere(diff > 0)
            starts += 1 #compensates for shifted different
            ends = numpy.argwhere(diff < 0)
            ends += 1 #compensates for shifted different
            if b[0]: starts = numpy.insert(starts, 0, 0)
            if b[-1]: ends = numpy.append(ends, len(a))

            z = numpy.zeros_like(a)

            if starts.size == 0 and ends.size == 0: return z

            for start, end in zip(starts, ends):
                maximum = numpy.argmax(a[int(start):int(end)])
                z[start+maximum] = a[start+maximum] * (end-start)

            return z


        # Horizontal direction
        for index in range(k.shape[0]):
            V[index,:] += _prob_1d(k[index,:,0])

        # Vertical direction
        for index in range(k.shape[1]):
            V[:,index] += _prob_1d(k[:,index,1])

        # Direction: 45 degrees (\)
        curv = k[:,:,2]
        i,j = numpy.indices(curv.shape)
        for index in range(-curv.shape[0]+1, curv.shape[1]):
            V[i==(j-index)] += _prob_1d(curv.diagonal(index))

        # Direction: -45 degrees (/)
        # NOTE: due to the way the access to the diagonals are implemented, in this
        # loop, we operate bottom-up. To match this behaviour, we also address V
        # through Vud.
        curv = numpy.flipud(k[:,:,3]) #required so we get "/" diagonals correctly
        Vud = numpy.flipud(V) #match above inversion
        for index in reversed(range(curv.shape[1]-1, -curv.shape[0], -1)):
            Vud[i==(j-index)] += _prob_1d(curv.diagonal(index))

        return V


    def connect_centres(self, V):
        """Connects vein centres by filtering vein probabilities ``V``

        This function does the equivalent of Step 2 / Equation 4 at Miura's paper.

        The operation is applied on a row from the ``V`` matrix, which may be
        acquired horizontally, vertically or on a diagonal direction. The pixel
        value is then reset in the center of a windowing operation (width = 5) with
        the following value:

          .. math::

             b[w] = min(max(a[w+1], a[w+2]) + max(a[w-1], a[w-2]))


        Parameters:

          V (numpy.ndarray): The accumulated vein centre probabilities ``V``. This
            is a 2D array with 64-bit floats and is defined by Equation (3) on the
            paper.


        Returns:

          numpy.ndarray: A 3-dimensional 64-bit array ``Cd`` containing the result
          of the filtering operation for each of the directions. ``Cd`` has the
          dimensions of $\kappa$ and $V_i$. Each of the planes correspond to the
          horizontal, vertical, +45 and -45 directions.

        """

        def _connect_1d(a):
            '''Connects centres in the given vector

            The strategy we use to vectorize this is to shift a twice to the left and
            twice to the right and apply a vectorized operation to compute the above.


            Parameters:

            a (numpy.ndarray): Input 1D array which will be window scanned


            Returns:

            numpy.ndarray: Output 1D array (must be writeable), in which we will
            set the corrected pixel values after the filtering above. Notice that,
            given the windowing operation, the returned array size would be 4 short
            of the input array.

            '''

            return numpy.amin([numpy.amax([a[3:-1], a[4:]], axis=0),
                               numpy.amax([a[1:-3], a[:-4]], axis=0)], axis=0)


        Cd = numpy.zeros(V.shape + (4,), dtype='float64')

        # Horizontal direction
        for index in range(V.shape[0]):
            Cd[index, 2:-2, 0] = _connect_1d(V[index,:])

        # Vertical direction
        for index in range(V.shape[1]):
            Cd[2:-2, index, 1] = _connect_1d(V[:,index])

        # Direction: 45 degrees (\)
        i,j = numpy.indices(V.shape)
        border = numpy.zeros((2,), dtype='float64')
        for index in range(-V.shape[0]+5, V.shape[1]-4):
          # NOTE: hstack **absolutately** necessary here as double indexing after
          # array indexing is **not** possible with numpy (it returns a copy)
            Cd[:,:,2][i==(j-index)] = numpy.hstack([border,
                                                    _connect_1d(V.diagonal(index)), border])

        # Direction: -45 degrees (/)
        Vud = numpy.flipud(V)
        Cdud = numpy.flipud(Cd[:,:,3])
        for index in reversed(range(V.shape[1]-5, -V.shape[0]+4, -1)):
          # NOTE: hstack **absolutately** necessary here as double indexing after
          # array indexing is **not** possible with numpy (it returns a copy)
            Cdud[:,:][i==(j-index)] = numpy.hstack([border,
                                                    _connect_1d(Vud.diagonal(index)), border])

        return Cd


    def binarise(self, G):
        """Binarise vein images using a threshold assuming distribution is diphasic

        This function implements Step 3 of the paper. It binarises the 2-D array
        ``G`` assuming its histogram is mostly diphasic and using a median value.


        Parameters:

          G (numpy.ndarray): A 2-dimensional 64-bit array ``G`` containing the
            result of the filtering operation. ``G`` has the dimensions of the
            original image.


        Returns:

          numpy.ndarray: A 2-dimensional 64-bit float array with the same
          dimensions of the input image, but containing its vein-binarised version.
          The output of this function corresponds to the output of the method.

        """

        median = numpy.median(G[G>0])
        Gbool = G > median
        return Gbool.astype(numpy.float64)


    def _view_four(self, k, suptitle):
        '''Display four plots using matplotlib'''

        import matplotlib.pyplot as plt

        k[k<=0] = 0
        k /= k.max()

        plt.subplot(2,2,1)
        plt.imshow(k[...,0], cmap='gray')
        plt.title('Horizontal')

        plt.subplot(2,2,2)
        plt.imshow(k[...,1], cmap='gray')
        plt.title('Vertical')

        plt.subplot(2,2,3)
        plt.imshow(k[...,2], cmap='gray')
        plt.title('+45 degrees')

        plt.subplot(2,2,4)
        plt.imshow(k[...,3], cmap='gray')
        plt.title('-45 degrees')

        plt.suptitle(suptitle)
        plt.tight_layout()
        plt.show()


    def _view_single(self, k, title):
        '''Displays a single plot using matplotlib'''

        import matplotlib.pyplot as plt

        plt.imshow(k, cmap='gray')
        plt.title(title)
        plt.tight_layout()
        plt.show()


    def __call__(self, image):

        finger_image = image[0].astype('float64')
        finger_mask = image[1]

        #import time
        #start = time.time()

        kappa = self.detect_valleys(finger_image, finger_mask)

        #self._view_four(kappa, "Valley Detectors - $\kappa$")

        #print('filtering took %.2f seconds' % (time.time() - start))
        #start = time.time()

        V = self.eval_vein_probabilities(kappa)

        #self._view_single(V, "Accumulated Probabilities - V")

        #print('probabilities took %.2f seconds' % (time.time() - start))
        #start = time.time()

        Cd = self.connect_centres(V)

        #self._view_four(Cd, "Connected Centers - $C_{di}$")
        #self._view_single(numpy.amax(Cd, axis=2), "Connected Centers - G")

        #print('connections took %.2f seconds' % (time.time() - start))
        #start = time.time()

        mc = self.binarise(numpy.amax(Cd, axis=2))

        #self._view_single(retval, "Final Binarised Image")

        #print('binarization took %.2f seconds' % (time.time() - start))
        
        
        
        img_veins_bin=np.zeros([1,1,mc.shape[0]-10,mc.shape[1]-10])
        img_veins_bin[0,0,:,:]=mc[5:-5,5:-5]#/255.
        img_veins_bin_ = torch.tensor(img_veins_bin, requires_grad=False).float().cuda()
        
        retval = model.get_embeding(img_veins_bin_).cpu().detach().numpy()
        
        return retval


extractor=MaximumCurvature()







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
            database.protocol = 'wrist-R_1'
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