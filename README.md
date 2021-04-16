Tomography code based on the Fast Adaptive Shrinkage Thresholding Algorithm (FASTA)
===
This tomography code is based on the [FASTA](http://www.cs.umd.edu/~tomg/projects/fasta/) implementation of the Fast Forward-Backward Splitting algorithm and the ASTRA library for forward and backward ray-transform operators. [The FASTA paper](https://arxiv.org/abs/1411.3406) talks about all the implementation details like adaptive stepsizes, accelerated convergence, and so on. 

Installation
============
- install [ASTRA](http://www.astra-toolbox.com/docs/install.html) from source to enable GPU capabilities
- install [pytorch](https://pytorch.org/) (tested with pytorch v1.5 and 1.6)
- install the Operator Discretization Library [ODL](https://github.com/odlgroup/odl) conda install -c odlgroup odl
- 'python setup.py install' in this folder
- [optional] 'python setup.py develop' for developer mode

Getting started
===========
- run ./tests/test_kates_data.py or ./tests/test_kates_data_multiscale.py


Possible coding projects to get up to speed with the code base
============
- interface Total General Variation regularization from [the CCPi regularisation toolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit) and write a Plug-and-Play regularizer to use in FASTA 
- port subpixel registration to the GPU, this is probably easiest with the CuPy package, since it has exactly the same interface as numpy. The code is in 
./fastatomography/util/register_translation.py, and is copied from skimage

TODO (in no particular order)
============
- implement higher-order TV regularisation, see above
