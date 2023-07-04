This set of codes are used for the refinment of FePt nanoparticle atomic model, published in Yang et al., Nature 542, 75 (2017).

Please see the "method" section of the paper [Xu et al., "Three-dimensional coordinates of individual atoms in materials revealed by electron tomography", Nature Materials 14, 1099-1103 (2015)] for the description of the refinement algorithm.

Most of the input data for this script is either included in this package, but some of them (ex. denoised projections and final refined Euler angles) can be separately downloaded from our group website (http://www.physics.ucla.edu/research/imaging/FePt).

XYZ_Refinement_Main.m is the main function to start with.

For the refined atomic structure, we monitored the err2 (saved as err2_arr_new.mat from the script) and found the configuration (i.e., atomic coordinates in pos_arr, saved in pos_arr_new.mat) which shows the minimum err2.


Author: Yongsoo Yang
email: yongsoo.ysyang@gmail.com (YY)
Jianwei (John) Miao Coherent Imaging Group
University of California, Los Angeles
Copyright (c) 2015. All Rights Reserved.