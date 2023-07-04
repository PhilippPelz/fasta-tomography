from genfire import main
from genfire.reconstruct import ReconstructionParameters, DisplayFigure
import os

# from plot import *

path = '/home/philipp/projects2/tomo/2019-03-18_Pd_loop/'
#path = 'C:\\Users\\Philipp\\Box\\ScottLab\\Data\\Xiaohui-2018\\2018-AuAg alloy nanowire project-original data\\20180703-Pd coating-tomo\\Reconstructions and Alignments'

filename_projections = path + 'data_raw_manual_bin.mat'  # filename of projections, which should be size NxNxN_projections where N_projections is the number of projections
filename_angles = path + 'angles.mat'  # angles can be either a 1xN_projections array containing a single tilt series, or 3xN_projections array containing 3 Euler angles for each projections in the form [phi;theta;psi]
filename_support = None  # '../support60.mat'  #NxNxN binary array specifying a region of 1's in which the reconstruction can exist
filename_initialObject = None  # initial object to use in reconstruction; set to None to provide no initial guess
filename_results = path + 'philipp_rec.mrc'  # filename to save results
resolutionExtensionSuppressionState = 2  # 1) Turn on resolution extension/suppression, 2) No resolution extension/suppression, 3) Just resolution extension

#filename_projections = path + '\\alignedProjections.mrc'  # filename of projections, which should be size NxNxN_projections where N_projections is the number of projections
#filename_angles = path + '\\angles.mat'  # angles can be either a 1xN_projections array containing a single tilt series, or 3xN_projections array containing 3 Euler angles for each projections in the form [phi;theta;psi]
#filename_support = None  # '../support60.mat'  #NxNxN binary array specifying a region of 1's in which the reconstruction can exist
#filename_initialObject = None  # initial object to use in reconstruction; set to None to provide no initial guess
#filename_results = path + '\\philipp_rec.mrc'  # filename to save results
#resolutionExtensionSuppressionState = 2  # 1) Turn on resolution extension/suppression, 2) No resolution extension/suppression, 3) Just resolution extension

# projections = loadProjections(filename_projections)  # load projections into a 3D numpy array
#print(projections.shape)
#projections = rebin(projections, (2, 2, 1))
#print(projections.shape)
numIterations = 50  # number of iterations to run in iterative reconstruction
oversamplingRatio = 2  # input projections will be padded internally to match this oversampling ratio. If you prepad your projections, set this to 1
interpolationCutoffDistance = 0.7  # radius of spherical interpolation kernel (in pixels) within which to include measured datapoints
doYouWantToDisplayFigure = False

displayFigure = DisplayFigure()
displayFigure.DisplayFigureON = doYouWantToDisplayFigure
displayFigure.DisplayErrorFigureON = False
displayFigure.displayFrequency = 10
calculateRFree = True

if filename_support is None:
    useDefaultSupport = True
else:
    useDefaultSupport = False

reconstruction_parameters = ReconstructionParameters()
reconstruction_parameters.projections = filename_projections
reconstruction_parameters.eulerAngles = filename_angles
reconstruction_parameters.support = filename_support
reconstruction_parameters.interpolationCutoffDistance = interpolationCutoffDistance
reconstruction_parameters.numIterations = numIterations
reconstruction_parameters.oversamplingRatio = oversamplingRatio
reconstruction_parameters.displayFigure = displayFigure
reconstruction_parameters.calculateRfree = calculateRFree
reconstruction_parameters.resolutionExtensionSuppressionState = resolutionExtensionSuppressionState
reconstruction_parameters.useDefaultSupport = useDefaultSupport
if os.path.isfile(filename_results):  # If a valid initial object was provided, use it
    reconstruction_parameters.initialObject = filename_results

main.main(reconstruction_parameters)
