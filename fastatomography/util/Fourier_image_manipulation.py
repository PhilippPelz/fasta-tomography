import numpy as np


def Fourier_shear(image,shear_factor,axis=[-1,-2],fftshifted=False):
    """Accomplishes the following affine transformation to an image:
        [x']  = [     1          shear_factor] [x]
        [y']    [     0               1      ] [y]
       via Fourier transform based methods.

       Parameters
       ----------
       image        -- Array like object containing image to be shear 
                       transformed
       shear_factor -- Amount to shear image by
       axis         -- List like object specifying axes to apply shear 
                       operation to. The axis[0] will be sheared relative
                       to axis[1]
       fftshifted   -- True if the object is already fft shifted, ie the
                       zeroth coordinate is in the top left hand corner
       """

    #If array is not a complex numpy array, make it so
    complex = np.iscomplexobj(image)
    if(not complex): image = np.asarray(image,dtype=np.csingle)
    
    #Pad output image to fit transformed array
    padding = np.zeros([len(image.shape),2],dtype=np.int)
    padding[axis[0],:] = np.ceil(shear_factor*image.shape[axis[1]]).astype(np.int)//2
    print(image.shape,shear_factor*image.shape[axis[1]])
    # print(shear_factor*image.shape[axis[0]],image.shape[axis[1]],image.shape[axis[0]])
    image_ = np.pad(image,padding,mode='constant')

    #Routine assumes that the origin is the top-left hand pixel
    if not fftshifted: image_ = np.fft.fftshift(image_,axes =axis)
    
    #Get shape of image
    Y,X = [image_.shape[i] for i in axis]
    print(Y,X)
    qy = np.fft.fftfreq(Y)
    qx = np.fft.fftfreq(X)

    #Fourier space shear operator
    a = np.exp(-2*np.pi*1j*qy[:,np.newaxis]*qx[np.newaxis,:]*X*shear_factor)
    
    #This function is designed to shear arbitrary axes
    #TODO test this functionality
    #This next part shapes the Fourier space shear
    #operator for appropriate broadcasting to the 
    #input arary
    ashape = ()
    ndim = len(image_.shape)
    for idim,dim in enumerate(image_.shape):
        if(idim == axis[0]%ndim or idim == axis[1]%ndim): ashape += (dim,)
        else: ashape +=(1,)
    
    if(axis[0]>axis[1]):
        a = a.T.reshape(ashape)
    else:
        a = a.reshape(ashape)
    # fig,ax=plt.subplots()
    # ax.imshow(a.real)
    # plt.show()
    #Apply shear operator
    image_ = np.fft.ifft(a*np.fft.fft(image_,axis=axis[0]),axis=axis[0])

    #Inverse FFT shift
    if not fftshifted: image_ = np.fft.ifftshift(image_,axes =axis)

    #Return same type as input
    if complex:
        return image_
    else:
        return np.real(image_)



def Fourier_rotate(image,theta,fftshifted=False,outsize='minimum'):
    """Performs a Fourier rotation of array image counterclockwise by angle
       theta in radians. Outsize can be 'original', 'minimum', 'double'

       Example
       ----------------------------------
       from skimage import data
       image = np.sum(np.asarray(data.astronaut()),axis=2)
       y,x = image.shape
       # image = np.pad(image,(y,x),'constant')
       fig,axes = plt.subplots(nrows=4,ncols=4,figsize=(16,16),squeeze=False)

       for i in range(16):
           image = np.sum(np.asarray(data.astronaut()),axis=2)
           # image = np.pad(image,(y,x),'constant')
           image = Fourier_rotate(image,np.deg2rad(360/16*(i)-45),outsize='original')
           axes[i//4,i%4].matshow(image)
           axes[i//4,i%4].set_axis_off()
           axes[i//4,i%4].set_title('{0} degrees'.format(360/16*(i)-45))
       plt.show()

       """

    #If array is not complex, make it so
    complex = np.iscomplexobj(image)
    if(not complex): image = np.asarray(image,dtype=np.csingle)

    #Fourier rotation only works for an angle less than 45, use np.rot90 to get
    #correct quadrant
    quadrant = np.round(theta/(np.pi/2))
    image = np.rot90(image,quadrant,axes=(-1,-2))

    iy,ix = image.shape[-2:]
    #Pad array by factor 2
    padding = np.zeros([len(image.shape),2],dtype=np.int)
    padding[-2,:] = iy//2
    padding[-1,:] = ix//2
    image = np.pad(image,padding,mode='constant')

    #Routine assumes that the origin is the top-left hand pixel
    if not fftshifted: image = np.fft.fftshift(image,axes =[-2,-1])

    #...and then Fourier rotation to desired angle within that quadrant
    fftrot = theta - quadrant*(np.pi/2)

    #Get shape of image
    Y,X = image.shape[-2:]
    qy = np.fft.fftfreq(Y)
    qx = np.fft.fftfreq(X)

    #Fourier space y shear operator
    a = np.exp(-2*np.pi*1j*qy[:,np.newaxis]*qx[np.newaxis,:]*Y*np.tan(fftrot/2))
    #Fourier space x shear operator
    b = np.exp( 2*np.pi*1j*qx[np.newaxis,:]*qy[:,np.newaxis]*X*np.sin(fftrot))

    #X shear
    image = np.fft.ifft(a*np.fft.fft(image,axis=-1),axis=-1)
    #Y shear
    image = np.fft.ifft(b*np.fft.fft(image,axis=-2),axis=-2)
    #X shear again
    image = np.fft.ifft(a*np.fft.fft(image,axis=-1),axis=-1)

    #Reset array coordinates to that of input
    if not fftshifted: image = np.fft.ifftshift(image,axes =[-2,-1])


    crop  = tuple([slice(0, i) for i in image.shape[:-2]])

    #Crop array to requested output size
    if outsize == 'original':
        crop += (slice(iy//2,-iy//2-iy%2),)
        crop += (slice(ix//2,-ix//2-ix%2),)
        image = image[crop]
    elif outsize == 'minimum':
        #Work output array size
        c, s = np.cos(theta), np.sin(theta)

        rot_matrix = np.array([[c, s],
                                 [-s, c]])

        # Compute transformed input bounds
        out_bounds = rot_matrix @ [[0, 0,ix, ix],
                                   [0,iy, 0, iy]]
        # Compute the shape of the transformed input plane
        out_plane_shape = (out_bounds.ptp(axis=1) + 0.5).astype(int)
        crop += (slice((Y-out_plane_shape[0])//2,(Y+out_plane_shape[0])//2),)
        crop += (slice((X-out_plane_shape[1])//2,(X+out_plane_shape[1])//2),)
        image = image[crop]

    #Return same type as input
    if complex:
        return image
    else:
        return np.real(image)
  

def fourier_interpolate_2d(ain,shapeout,norm=True):
    '''ain - input numpy array, npiy_ number of y pixels in
    interpolated image, npix_ number of x pixels in interpolated 
    image. Perfoms a fourier interpolation on array ain.'''
    #Import required FFT functions
    from numpy.fft import fftshift,fft2,ifft2

    #Make input complex
    aout = np.zeros(shapeout,dtype=np.complex)    

    #Get input dimensions
    npiyin,npixin = np.shape(ain)
    npiyout,npixout = shapeout

    #Construct input and output fft grids
    qyin,qxin,qyout,qxout  = [(np.fft.fftfreq(x,1/x ) ).astype(np.int) 
                              for x in [npiyin,npixin,npiyout,npixout]]
    
    #Get maximum and minimum common reciprocal space coordinates
    minqy,maxqy = [max(np.amin(qyin),np.amin(qyout)),min(np.amax(qyin),np.amax(qyout))]
    minqx,maxqx = [max(np.amin(qxin),np.amin(qxout)),min(np.amax(qxin),np.amax(qxout))]

    #Make 2d grids
    qqxout,qqyout = np.meshgrid(qxout,qyout)
    qqxin ,qqyin  = np.meshgrid(qxin ,qyin )

    #Make a masks of common Fourier components for input and output arrays
    maskin = np.logical_and(np.logical_and(qqxin <= maxqx,qqxin >= minqx),
                            np.logical_and(qqyin <= maxqy,qqyin >= minqy))
    
    maskout = np.logical_and(np.logical_and(qqxout <= maxqx,qqxout >= minqx),
                             np.logical_and(qqyout <= maxqy,qqyout >= minqy))
    
    #Now transfer over Fourier coefficients from input to output array
    aout[maskout] = fft2(np.asarray(ain,dtype=np.complex))[maskin]

    #Fourier transform result with appropriate normalization
    aout = ifft2(aout)
    if(norm): aout*=(1.0*np.prod(shapeout)/np.prod(np.shape(ain)))
    #Return correct array data type
    if (str(ain.dtype) in ['float64','float32','float','f'] or str(ain.dtype)[1] == 'f'): return np.real(aout)
    else: return aout



if __name__=="__main__":
    import matplotlib.pyplot as plt

    # Get our test image -- a cat, as distorting
    # the astronaut would be disrespectful
    from skimage.data import checkerboard,chelsea
    test_image = np.sum(chelsea(),axis=2).astype(np.float)
    test_image = checkerboard().astype(np.float)

    fig,ax = plt.subplots(nrows=3,ncols=2,figsize=(8,12))

    ax[0,0].imshow(test_image)
    ax[0,0].set_title('Original')
    ax[0,1].imshow(Fourier_shear(test_image,0.2,axis=[0,1]))
    ax[0,1].set_title('y shear')
    ax[1,0].imshow(Fourier_shear(test_image,0.2,axis=[1, 0]))
    ax[1,0].set_title('x shear')
    ax[1,1].imshow(Fourier_rotate(test_image,67))
    ax[1,1].set_title('rotation')
    ax[2,0].imshow(fourier_interpolate_2d(test_image,[x//4 for x in test_image.shape]))
    ax[2,0].set_title('Fourier downsampling')
    ax[2,1].imshow(fourier_interpolate_2d(test_image,[2*x for x in test_image.shape]))
    ax[2,1].set_title('Fourier upsampling')

    # for i in range(6): ax[i//2,i%2].set_axis_off()
    plt.tight_layout()
    plt.show()
    # fig.savefig('Image_manipulations_checkerboard.png')






