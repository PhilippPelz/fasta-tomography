# Atom Tracing Code for International Workshop and Short Course on the FRONTIERS OF ELECTRON TOMOGRAPHY
# https://www.electron-tomo.com/

import numpy as np
import scipy as sp
import scipy.io as sio
import os
import warnings


def tripleRoll(vol, vec):
    return np.roll(np.roll(np.roll(vol, vec[0], axis=0), vec[1], axis=1), vec[2], axis=2)

def peakFind3D(vol, thresh3D):
    """
    Find peaks in a 3D volume
    vol: an ndarray of values with peaks to find
    thresh3D: [0,1] value to set a threshold for size of peak vs. max intensity in image
    """
    pLarge = ((vol > tripleRoll(vol, [-1, -1, -1]))
              & (vol > tripleRoll(vol, [0, -1, -1]))
              & (vol > tripleRoll(vol, [1, -1, -1]))
              & (vol > tripleRoll(vol, [-1, 0, -1]))
              & (vol > tripleRoll(vol, [1, 0, -1]))
              & (vol > tripleRoll(vol, [-1, 1, -1]))
              & (vol > tripleRoll(vol, [0, 1, -1]))
              & (vol > tripleRoll(vol, [1, 1, -1]))
              & (vol > tripleRoll(vol, [0, 0, -1]))
              & (vol > tripleRoll(vol, [-1, -1, 0]))
              & (vol > tripleRoll(vol, [0, -1, 0]))
              & (vol > tripleRoll(vol, [1, -1, 0]))
              & (vol > tripleRoll(vol, [-1, 0, 0]))
              & (vol > tripleRoll(vol, [1, 0, 0]))
              & (vol > tripleRoll(vol, [-1, 1, 0]))
              & (vol > tripleRoll(vol, [0, 1, 0]))
              & (vol > tripleRoll(vol, [1, 1, 0]))
              & (vol > tripleRoll(vol, [-1, -1, 1]))
              & (vol > tripleRoll(vol, [0, -1, 1]))
              & (vol > tripleRoll(vol, [1, -1, 1]))
              & (vol > tripleRoll(vol, [-1, 0, 1]))
              & (vol > tripleRoll(vol, [1, 0, 1]))
              & (vol > tripleRoll(vol, [-1, 1, 1]))
              & (vol > tripleRoll(vol, [0, 1, 1]))
              & (vol > tripleRoll(vol, [1, 1, 1]))
              & (vol > tripleRoll(vol, [0, 0, 1]))
              & (vol > thresh3D * np.max(vol)))

    [xp, yp, zp] = np.where(pLarge * vol)
    ip = vol[xp, yp, zp]
    return {'xp': xp, 'yp': yp, 'zp': zp, 'ip': ip}


def MatrixQuaternionRot(vector, theta):
    """
    MatrixQuaternionRot(vector,theta)
    Returns a 3x3 rotation matrix [SO(3)] in numpy array (not numpy matrix!)
    for rotating "theta" angle around the given "vector" axis.

     vector - A non-zero 3-element numpy array representing rotation axis
     theta - A real number for rotation angle in "DEGREES"

     Author: Yongsoo Yang, Dept. of Physics and Astronomy, UCLA
             yongsoo.ysyang@gmail.com
    """
    theta = theta * np.pi / 180
    vector = vector / np.float(np.sqrt(np.dot(vector, vector)))
    w = np.cos(theta / 2)
    x = -np.sin(theta / 2) * vector[0]
    y = -np.sin(theta / 2) * vector[1]
    z = -np.sin(theta / 2) * vector[2]
    RotM = np.array([[1. - 2 * y ** 2. - 2 * z ** 2, 2. * x * y + 2 * w * z, 2. * x * z - 2. * w * y], \
                     [2. * x * y - 2. * w * z, 1. - 2. * x ** 2 - 2. * z ** 2, 2. * y * z + 2. * w * x], \
                     [2 * x * z + 2 * w * y, 2 * y * z - 2. * w * x, 1 - 2. * x ** 2 - 2. * y ** 2]])

    return RotM


def gauss3DGEN_FIT(xyz, x0, y0, z0, sigma_x, sigma_y, sigma_z, Angle1, Angle2, Angle3, BG, Height):
    """
    gauss3DGEN_FIT((x,y,z),x0,y0,z0,sigma_x,sigma_y,sigma_z,Angle1,Angle2,Angle3,BG,Height)
    Returns the value of a gaussian at a 3D set of points for the given
    sub-pixel positions with standard deviations, 3D Eular rotation angles,
    constant Background value, and Gaussian peak height.
     xyz - A tuple containing the 3D arrays of points (possibly from meshgrid)
     x0, y0, z0 = the x, y, z centers of the Gaussian
     sigma_x,sigma_y,sigma_z = standard deviations along x,y,z direction before
                               3D angular rotation
     Angle1,Angle2,Angle3 = Tiat-Bryan angles in ZYX convention for 3D rotation
     BG = constant background
     Height = the peak height of Gaussian function

     Author: Yongsoo Yang, Dept. of Physics and Astronomy, UCLA
             yongsoo.ysyang@gmail.com
    """
    # 3D vectors for each sampled positions
    v = np.array(
        [xyz[0].reshape(-1, order='F') - x0, xyz[1].reshape(-1, order='F') - y0, xyz[2].reshape(-1, order='F') - z0])

    # rotation axes for Tiat-Bryan angles
    vector1 = np.array([0, 0, 1])
    rotmat1 = MatrixQuaternionRot(vector1, Angle1)

    vector2 = np.array([0, 1, 0])
    rotmat2 = MatrixQuaternionRot(vector2, Angle2)

    vector3 = np.array([1, 0, 0])
    rotmat3 = MatrixQuaternionRot(vector3, Angle3)

    # full rotation matrix
    rotMAT = np.matrix(rotmat3) * np.matrix(rotmat2) * np.matrix(rotmat1)

    # 3x3 matrix for applying sigmas
    D = np.matrix(np.array([[1. / (2 * sigma_x ** 2), 0, 0, ], \
                            [0, 1. / (2 * sigma_y ** 2), 0], \
                            [0, 0, 1. / (2 * sigma_z ** 2)]]))

    # apply 3D rotation to the sigma matrix
    WidthMat = np.transpose(rotMAT) * D * rotMAT

    # caltulate 3D Gaussian
    RHS_calc = WidthMat * np.matrix(v)
    Result = Height * np.exp(-1 * np.sum(v * RHS_calc.A, axis=0)) + BG

    return Result


def saveXYZ(filename, xyzCoords, AtomTypes, AtomNames, DataDescriptionStr):
    """
    saveXYZ(filename,xyzCoords,AtomTypes,AtomNames,DataDescriptionStr)
    Writes the 3D atom coordinates and species into a file in xyz format.
     filename - filename to save the atomic coordinates
     xyzCoords - 3xN numpy array for xyz coordinates
     AtomTypes - postive integer describing the atomic species (0,1,2,...)
     AtomNames - a list of strings, containing the name of the atomic species
                 for each atomtype of corresponding AtomTypes
     DataDescriptionStr - a string to be written on the header

     Author: Yongsoo Yang, Dept. of Physics and Astronomy, UCLA
             yongsoo.ysyang@gmail.com
    """

    f = open(filename, 'w')
    f.write('{0:d}\n'.format(xyzCoords.shape[1]))
    f.write('{0:s}\n'.format(DataDescriptionStr))
    for i in range(xyzCoords.shape[1]):
        f.write('{0:s}     {1:10.5f} {2:10.5f} {3:10.5f}\n'.format(AtomNames[int(AtomTypes[i] - 1)], xyzCoords[0, i],
                                                                   xyzCoords[1, i], xyzCoords[2, i]))
    f.close()


def compute_average_atom_from_vol(DataMatrix, atom_pos, atom_ind, boxhalfsize):
    """
    compute_average_atom_from_vol(DataMatrix,atom_pos,atom_ind,boxhalfsize)
    Computes the average atom based on given DataMatrix and atom positions and
    indices with the boxhalfsize.
     DataMatrix - a 3D array for reconstructed 3D intensity
     atom_pos - a 3xN numpy array for xyz coordinates
     atom_ind - a 1D numpy array for indices of atoms to be averaged
     boxhalfsize - half box size for the cropping, full box size will be
                   (2*boxhalfsize+1,2*boxhalfsize+1,2*boxhalfsize+1)

     Author: Yongsoo Yang, Dept. of Physics and Astronomy, UCLA
             yongsoo.ysyang@gmail.com
    """

    total_vol = np.zeros((boxhalfsize * 2 + 1, boxhalfsize * 2 + 1, boxhalfsize * 2 + 1))

    for kkk in atom_ind:
        curr_x = int(np.round(atom_pos[0, kkk]))
        curr_y = int(np.round(atom_pos[1, kkk]))
        curr_z = int(np.round(atom_pos[2, kkk]))

        curr_vol = DataMatrix[curr_x - boxhalfsize:curr_x + boxhalfsize + 1, \
                   curr_y - boxhalfsize:curr_y + boxhalfsize + 1, \
                   curr_z - boxhalfsize:curr_z + boxhalfsize + 1]
        total_vol = total_vol + curr_vol

    avatom = total_vol / len(atom_ind)
    return avatom


def get_atomtype_vol_twoatom_useInd(DataMatrix, atom_pos, avatomFe, avatomPt, boxhalfsize, useInd):
    """
     Author: Yongsoo Yang, Dept. of Physics and Astronomy, UCLA
             yongsoo.ysyang@gmail.com
    """

    atomtype = np.zeros((atom_pos.shape[1]))
    numAtom1 = 0
    numAtom2 = 0
    for kkk in range(atom_pos.shape[1]):
        curr_x = int(np.round(atom_pos[0, kkk]))
        curr_y = int(np.round(atom_pos[1, kkk]))
        curr_z = int(np.round(atom_pos[2, kkk]))

        curr_vol = DataMatrix[curr_x - boxhalfsize:curr_x + boxhalfsize + 1, \
                   curr_y - boxhalfsize:curr_y + boxhalfsize + 1, \
                   curr_z - boxhalfsize:curr_z + boxhalfsize + 1]

        D_Fe = np.sum(np.abs(curr_vol[useInd] - avatomFe[useInd]))
        D_Pt = np.sum(np.abs(curr_vol[useInd] - avatomPt[useInd]))

        if D_Fe > D_Pt:
            atomtype[kkk] = 2
            numAtom2 += 1
        else:
            atomtype[kkk] = 1
            numAtom1 += 1

    # print('hi')
    # print("number of Fe/Pt atoms: {0:d}/{0:d}".format( numAtom1,numAtom2))
    # print("number of Pt: {0:d} atoms".format( sum(atomtype==2)))
    return atomtype


def trace_atoms(volume, GaussRad=2, ClassificationRad=2, Res=0.35, MinDist=2., intensityThresh=0.08):
    """Define this method for Python operators that
    transform the input array
    inputs:
    dataset - the volume to trace as a 3D numpy array
    GaussRad - pixel radius for cropping small volume around each peak for Gaussian fitting
    ClassificationRad - radius for cropping small volume around a peak for atom classification
    Res - pixel resolution (isotropic)
    MinDist - enforce minimum distance between atom positions
    intensityThresh - threshold to find peaks in the volume

    """

    warnings.filterwarnings('ignore')
    print('Start atom tracing.')

    # GaussRad = 2 # radius for cropping small volume for fitting Gaussian
    # Res = 0.35  # pixel size
    MinDist = MinDist / Res  # minimum distance in Angstrom
    # ClassificationRad = 2 # radius for cropping small volume for species classification

    t1_vol1 = volume.copy()
    t1_vol1 = np.swapaxes(t1_vol1, 0, 2)

    pp3D = peakFind3D(t1_vol1, intensityThresh)  # volume and intensity threshold
    allPeaksIXYZ = np.array((pp3D['ip'], pp3D['xp'], pp3D['yp'], pp3D['zp'])).T

    print("Number of peaks found: {}".format(allPeaksIXYZ.shape[0]))

    sortedPeaksIXYZ = allPeaksIXYZ[allPeaksIXYZ[:, 0].argsort()[::-1]]
    RecordedPeakPos = np.array([])
    fittingValues = np.zeros((sortedPeaksIXYZ.shape[0], 11))

    print('Start Gaussian fitting for each peak.')
    cutOut3 = GaussRad
    Y3D, X3D, Z3D = np.meshgrid(np.arange(-cutOut3, cutOut3 + 1, 1), np.arange(-cutOut3, cutOut3 + 1, 1),
                                np.arange(-cutOut3, cutOut3 + 1, 1))
    for PeakInd in range(sortedPeaksIXYZ.shape[0]):
        curX = sortedPeaksIXYZ[PeakInd, 1]
        curY = sortedPeaksIXYZ[PeakInd, 2]
        curZ = sortedPeaksIXYZ[PeakInd, 3]

        # Check if current point + fit region is within the bounds of the volume
        if (curX > cutOut3 + 1) & (curY > cutOut3 + 1) & (curZ > cutOut3 + 1) & (curX < t1_vol1.shape[0] - cutOut3) \
                & (curY < t1_vol1.shape[1] - cutOut3) & (curZ < t1_vol1.shape[2] - cutOut3):

            curVol = np.float32(t1_vol1[int(curX) - cutOut3:int(curX) + cutOut3 + 1, \
                                int(curY) - cutOut3:int(curY) + cutOut3 + 1, \
                                int(curZ) - cutOut3:int(curZ) + cutOut3 + 1])

            # Fit to a 3D Gaussian to the area around the peak
            initP3D = (0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0., 0., 0., 0.,
                       max(curVol.flatten()))  # x0,y0,z0,sigma_x,sigma_y,sigma_z,angle1, angle2, angle3,BG,Height
            bb = (
            (-cutOut3, -cutOut3, -cutOut3, 0.0, 0.0, 0.0, -np.pi, -np.pi, -np.pi, -np.inf, max(curVol.flatten()) / 2.), \
            (cutOut3, cutOut3, cutOut3, cutOut3 * 3, cutOut3 * 3, cutOut3 * 3, np.pi, np.pi, np.pi, np.inf,
             max(curVol.flatten()) * 2.))  # fitting bounds ((lower),(upper))
            optP3D, optCov3D = sp.optimize.curve_fit(gauss3DGEN_FIT, (X3D, Y3D, Z3D), curVol.reshape(-1, order='F'),
                                                     p0=initP3D, bounds=bb)

            # Check minimum distance constraint and add to the array if good
            fittingValues[PeakInd, :] = optP3D
            if RecordedPeakPos.size == 0:  # first peak, no need to check minimum dist
                RecordedPeakPos = np.array([[curX + optP3D[0]], [curY + optP3D[1]], [curZ + optP3D[2]]])
                # print("recorded atom number 1, peak number 1")
            else:
                Dist = np.sqrt(np.sum(
                    (RecordedPeakPos - np.array([[curX + optP3D[0]], [curY + optP3D[1]], [curZ + optP3D[2]]])) ** 2,
                    axis=0))
                if np.min(Dist) >= MinDist:
                    RecordedPeakPos = np.append(RecordedPeakPos,
                                                [[curX + optP3D[0]], [curY + optP3D[1]], [curZ + optP3D[2]]], axis=1)
                    # print("recorded atom number {0:d}, peak number {1:d}".format(RecordedPeakPos.shape[1],PeakInd+1))
                # else:
                # print("peak number {0:d} rejected due to minimum distance violation".format(PeakInd))

    print('Start atom classification based on peak intensity.')
    intensity_integ = np.zeros((RecordedPeakPos.shape[1]))
    boxhalfsize = 1

    for j in range(RecordedPeakPos.shape[1]):
        curr_x = int(np.round(RecordedPeakPos[0, j]))
        curr_y = int(np.round(RecordedPeakPos[1, j]))
        curr_z = int(np.round(RecordedPeakPos[2, j]))

        curr_vol = t1_vol1[curr_x - boxhalfsize:curr_x + boxhalfsize + 1, \
                   curr_y - boxhalfsize:curr_y + boxhalfsize + 1, \
                   curr_z - boxhalfsize:curr_z + boxhalfsize + 1]

        intensity_integ[j] = np.sum(curr_vol)

    I_sorted = np.sort(intensity_integ)
    TH1 = I_sorted[int(np.round(len(I_sorted) / 2.))]

    defFeind = np.where(intensity_integ < TH1)
    defPtind = np.where(intensity_integ >= TH1)

    avatomFe = compute_average_atom_from_vol(t1_vol1, RecordedPeakPos, defFeind[0], ClassificationRad)
    avatomPt = compute_average_atom_from_vol(t1_vol1, RecordedPeakPos, defPtind[0], ClassificationRad)

    atomtype = np.zeros((len(intensity_integ)))
    atomtype[defFeind] = 1
    atomtype[defPtind] = 2

    # %% atom classification iteration of Fe and Pt
    curr_iternum = 0
    exitFlag = 0

    while exitFlag == 0:
        # print('Atom classification iteration {0:d}'.format(curr_iternum))

        old_atomtype = atomtype.copy()

        # % obtain updated atom classification by comparing each peaks with average atom species
        atomtype = get_atomtype_vol_twoatom_useInd(t1_vol1, RecordedPeakPos, avatomFe, avatomPt, ClassificationRad,
                                                   np.where(curr_vol > -np.inf)[0])

        # % re-compute average atomic species from the updated atom classification
        avatomFe = compute_average_atom_from_vol(t1_vol1, RecordedPeakPos, np.where(atomtype == 1)[0],
                                                 ClassificationRad)
        avatomPt = compute_average_atom_from_vol(t1_vol1, RecordedPeakPos, np.where(atomtype == 2)[0],
                                                 ClassificationRad)

        curr_iternum = curr_iternum + 1

        # % if there is no change in the atomic specise classification, turn on
        # the exitflag
        if sum(old_atomtype != atomtype) == 0:
            exitFlag = 1

    # saveXYZ(os.path.expanduser('~\Desktop\AET_finalXYZ.xyz'),RecordedPeakPos*Res,atomtype,['Fe','Pt'],"FePt test")

    outName = os.path.normpath(os.path.expanduser('~/Desktop/AET_finalXYZ.xyz'))
    saveXYZ(outName, RecordedPeakPos * Res, atomtype, ['Fe', 'Pt'], "FePt test")
    print('Data saved to {}'.format(outName))
    print('Finished atom tracing.')
