import matplotlib as mpl
from numpy.random import uniform
from scipy.ndimage import fourier_shift
from skimage.feature import register_translation
from tqdm import trange

# mpl.rcParams['text.usetex'] = True
from fastatomography.fasta_tomography import fasta_tomography_nonnegative_shrink, fasta_tomography_nonnegative_shrink_support_constraint
from fastatomography.default_dependencies import *
from scipy.io import loadmat
from skimage.filters import gaussian
from fastatomography.util import *
from fastatomography.tomo import ray_transforms
from fastatomography.util.plotting import plot_rotating_point, plot_translations, save_stack_movie, save_stack_gif

import han_utils as h

# Set paths and some parameters
path = '/home/hannah/data/Mengyu/'
pathout = '/home/hannah/data/Mengyu/results_support/'
fig_path = pathout
fn = '400px_aligned.mat'
d = 'stack4'
angles_fn = 'angles.mat'
a = 'angles'
sup = '400px_support.mat'
select1 = 'c'
dx = .27  # 1 #34.76e-2 / 2     # How finely sampled we want our fourier coordinates to be (can set this to the pixelsize (in Angstroms) of the input images)
resolution_cutoff = 0.5         # Lowest resolution to create
resolutions = np.array([2.5, 1, resolution_cutoff])  # Array of resolutions we want too look at
pplot = False  # Plot as we go/ save images

# Set the GPU device to use
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
#th.cuda.set_device(device)

# Load data
proj = loadmat(path + fn)[d]
support = loadmat(path + sup)[select1]
#support = th.as_tensor(support, dtype=th.bool).contiguous().to(device)
angles  = h.Angles(path+angles_fn, a )  # Load angles and put into correct shape

class Recon():
    
    def __init__( self, projections, angles, pixelsize, pathout, fig_path, resolution_cutoff=0.5, recon_resolution=0.5, recon_crop=160, opts_r=None,
                  name=None, opts_a=None, support=None, pplot=False , angle_refine_opts=None, mu_best=1, angle_refine_crop=None):
        self.angles             = angles
        self.n_angles           = angles.n_angles
        self.dx                 = pixelsize
        self.recon_resolution   = recon_resolution
        self.resolution_cutoff  = resolution_cutoff
        self.support            = support
        self.pathout            = pathout
        self.figpath            = fig_path
        self.recon_crop         = recon_crop
        self.mu_best            = mu_best
        if not name:
            self.name = 'reconstruction'
                #self.upsample_factors  = resolutions / resolution_cutoff
                #self.resolution_ratios = self.upsample_factors*1

        if not opts_r:
            opts = Param()
            opts.record_objective = False
            opts.verbose = True
            opts.string_header = '     '
            opts.max_iters = 200
            opts.tol = 1e-5
            opts.accelerate = False
            opts.adaptive = False
            opts.restart = True
            opts.tau = 20
        
        self.opts_r = opts_r
        self.angle_refine_opts = angle_refine_opts
        self.angle_refine_crop = angle_refine_crop

        self.reshape_pad_projections( projections )
        self.get_fourier_pixelsize()
        self.get_fpstack()
        self.make_2x_full_res( self.fpstack )

    def do_a_reconstruction( self, angle_refine ):
        '''Perform a reconstruction. If angle_refine=None no angle refinement will be done. If angle_refine=a resolution, then that resolution will be used to
        do angle refinement. If angle_refinement=True, then angle refinement will be done at a default of 0.5x resolution (res=2).'''
        if angle_refine:
            if isinstance(angle_refine, bool):
                angle_refine = 2
            res_A = Res(angle_refine, self.dq, self.fpstack_fullres, self.angles) # Get the resolution wanted for Angle refinement
            res_A.normalize_pstack( self.pstack , self.n_angles )  # Normalize resolution stack to our input projections
            if self.angle_refine_crop is not None:
                res_A.crop( self.angle_refine_crop, pf='pstack' )
            angleRef = AngleRefinement( res_A, self.angle_refine_opts, iterations=2 )
            angleRef.mu_best = self.mu_best
            angleRef.refine()
            self.translation_shifts = angleRef.translation_shifts
            self.translation_shifts_resfactor = res_A.res

        # Pick a new resolution to do the final reconstruction at
            res_R = Res(self.recon_resolution, self.dq, self.fpstack_fullres, self.angles ) #!! Creates a second copy of the full_res, not on GPU. TODO: Find a way to just refer to fpstack_fullres if needed
            res_R.normalize_pstack( self.pstack, self.n_angles )
        
        # Shift the stack according to the best shifts found in the angle refinement, crop to get ready for reconstruction 
        res_R.shift( translation_shifts=self.translation_shifts, translations_upsample_factor=self.translation_shifts_resfactor )
        res_R.crop( self.recon_crop, pf='pstack', keep_original=True ) # Crop pstack to the right size for reconstruction
        #self.plot_mosaic( res_R.pstack, 'Recon Stack: Shifted and Cropped', 'Rstack_shift_crop_tile', pplot=True )
        plot( np.sum(res_R.pstack,0), 'Sum(Recon Stack)' )
        sol, out, opts_out = self.reconstruct( res_R, angleRef.best_angles, self.opts_r, self.mu_best, pplot=True )
        np.save(self.pathout + self.name, sol.cpu().numpy())
    
    def reconstruct(self, res_obj, angles, opts, mu, pplot=False ):
        
        print("Reconstructing...")#f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing ...')
        projection_shape = res_obj.N_pstack[1:]
        vol_shape = (projection_shape[0], projection_shape[1],projection_shape[1])
        real_space_extent = np.array([projection_shape[0], projection_shape[0], projection_shape[1]])
        A, At = ray_transforms(real_space_extent, projection_shape, res_obj.n_angles, interp='linear')
        x0 = th.zeros(*vol_shape, dtype=th.float32).to(device)
        y = th.as_tensor(np.transpose(res_obj.pstack, (1, 0, 2))).contiguous().float().to(device)
        th.cuda.empty_cache()
        if self.support:
            support = th.as_tensor(self.support,dtype=th.bool).no_grad().contiguous().cpu() #.to(device)
            sol, out, opts_out = fasta_tomography_nonnegative_shrink_support_constraint(A, At, x0, y, mu, angles, support, device, opts)
        else:
            sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, angles, opts)
        
        print(f'best R-factor: {out.R_factors[-1]}')
        # Plot if requested
        if pplot:
            self.plot_results( sol, out, res_obj, y, angles, A )
        
        print("Done with reconstruction")
        return sol, out, opts_out
    
    def plot_results( self, sol, out, res_stack, y, angles, A ):

        mp = th.zeros_like(y)
        mp = A(sol, out=mp, angles=angles)

        print(res_stack.N)
        i = int(np.ceil(res_stack.N_pstack[0] / 2))
        plot(np.hstack([y[:, i, :].cpu(), mp[:, i, :].cpu()]), 'Projections           Model_projections')

        diff = np.transpose((y - mp).cpu().numpy(), (1, 0, 2))
        plotmosaic(diff, 'y - y_model', dpi=900)

        rec = sol.cpu().numpy()
        f, a = plt.subplots(1, 3, dpi=300)
        a[0].imshow(rec[res_stack.N_pstack[1] // 2, :, :])
        a[0].set_xlabel(f'z')
        a[0].set_ylabel(f'x')
        a[1].imshow(rec[:, res_stack.N_pstack[1]// 2, :])
        a[1].set_xlabel(f'z')
        a[1].set_ylabel(f'y')
        a[2].imshow(rec[:, :, res_stack.N_pstack[1]// 2])
        a[2].set_xlabel(f'x')
        a[2].set_ylabel(f'y')
        plt.show()
        f.savefig(f'{self.figpath}xyz_fullres.png', dpi=300)
        # %%
        f, a = plt.subplots(1, 3, dpi=300)
        a[0].semilogy(out.residuals)
        a[0].set_title(r"$||x_n-x_{n+1}||^2$")
        a[1].semilogy(out.objective)
        a[1].set_title(r"$||y-Ax||^2$")
        a[2].semilogy(out.R_factors)
        a[2].set_title(r"R-factor $=\frac{||y-Ax||_1}{||y||_1}$")
        plt.show()
        f.savefig(f'{self.figpath}residuals_objective__fullres.png', dpi=300)


    def reshape_pad_projections( self , projections, pad=None, gif_out=False):
        # Reshape projections
        angle_dim = np.where( np.array(projections.shape) == self.n_angles )[0][0]
        if angle_dim == 0:
            pstack = projections
        elif angle_dim == 1:
            pstack = np.transpose(projections, (1, 0, 2) )
        elif angle_dim == 2:
            pstack = np.transpose(projections, (2, 0, 1) )
        
        #self.plot_mosaic( pstack, 'Tilt series overview', 'series_overview', pplot=False )
        
        # Pad in real space to alleviate aliasing artifacts
        N = np.array(pstack.shape[1:])
        if not pad:
            pad = N[0] // 4 
        pstack = np.pad(pstack, ((0, 0), ( pad, pad), (pad, pad)), mode='constant', constant_values=0)
            
        self.n_proj = pstack.shape[0]
        self.N      = np.array( pstack.shape[1:] ) # Shape of the 
        self.pstack = pstack

        if gif_out:
            save_stack_gif(self.pathout + 'gif', pstack, np.rad2deg(angles.angles), self.dx )
    
    def get_fourier_pixelsize( self ):
        q  = fourier_coordinates_2D(self.N, [self.dx, self.dx], centered=False) # Make fourier coordinates (complex and real)
        qn = np.linalg.norm(q, axis=0)   # Take norm to get the difference between pixels in 1/angstroms
        self.dq   = qn[0, 1]     # Get pixelsize in 1/angstroms

    def get_resolution_cutoff_mask( self, res, fullres_size=None, falloff=20, pplot=False):
        """ Create a fourier cutoff mask for cutoff resolution."""
        if not fullres_size:
            try:
                fullres_size = self.N_fullres
            except AttributeError as a:
                raise Exception(str(a) + 'Please specify the size of the full-resolution stack that you want to create a mask for by calling get_resolution_cutoff_mask with the fullres_size option, or add the size to the Recon instance under N_fullres.')
        qcutoff = 1 / (2 * res)
        rcutoff = qcutoff / self.dq       # Get the cutoff frequency in fourier pixels (cutoff frequency will be half of the total number of pixels associated with the fourier transform)
        cutoff_resolution_mask = sector_mask(self.N_fullres, self.N_fullres // 2, rcutoff, (0, 360)).astype(np.float32) # Create a fourier mask for the cutoff resolution
        cutoff_resolution_mask = gaussian(cutoff_resolution_mask, falloff)  # Window the mask so that it's not a stark cutoff
    
        if pplot:
            fig, ax = plt.subplots(dpi=150)
            im = ax.imshow(avg_ps * cutoff_resolution_mask, interpolation='nearest', cmap=plt.cm.get_cmap('inferno'))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_title('log10(var(abs(fft2(stack))**2))')
            ax.grid(False)
            for r1, res in zip(r, resolutions):
                circle1 = plt.Circle(N / 2, r1, color='b', fill=None)
                ax.add_artist(circle1)
                txt = ax.text(*(N // 2 - r1), f'{res:2.2f} A', color='b')

            plt.show()
            fig.savefig(f'{fig_path}log_var.png', dpi=300)
        self.cutoff_resolution_mask = cutoff_resolution_mask
    
    def make_2x_full_res(self, fpstack):
        """Pad to 2x full resolution, so that maximum resolved lattice spacing has 4 pixels instead of 2.
        Returns a fourier volume which is cutoff at the Nyquist frequency"""
        r = np.ceil( 1 / self.dq ) # r is the inverse fourier-pixelsize which I think should be the original pixel size
        max_size = np.array([4 * r, 4 * r], dtype=np.int) # The max size can be
        pad = (max_size - self.N) // 2
        pad[pad < 0] = 0
        padding = ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]))
        fpstack_fullres = fftshift(np.pad(fftshift(fpstack, (1, 2)), padding, mode='constant', constant_values=0),
                                   (1, 2))  # Pad fourier stack to four times the original size

        self.N_fullres = np.array(fpstack_fullres.shape[1:])
        self.get_resolution_cutoff_mask( self.resolution_cutoff, falloff=20, pplot=pplot )

        fpstack_fullres *= fftshift(self.cutoff_resolution_mask)
        self.fpstack_fullres = fpstack_fullres
   # def plot_mosaic( self, stack, title='', save='default', dpi=600, pplot=True ):
   #     if pplot:
   #         plotmosaic(stack, title, dpi=dpi, savePath=self.figpath + save )

    def get_fpstack(self, gif=True):
        '''Take fourier transform, calculate the variance of the f-space projections, 
        get the average of the variance'''
        self.fpstack = fft2(self.pstack, norm='ortho')
        self.fpstack_variance = np.var(fftshift(np.abs(self.fpstack) ** 2), 0)  #ps Get the variance of the projections in fourier space with respect to each other
        avg_ps = np.log10( self.fpstack_variance )
        self.Nfp = self.fpstack.shape[1:] 

class Res():
    ''' Creates a fourier-space stack at resolution res, from the full res stack. Has functions for normalizing the stack to a separate input stack, plotting a sum of the stack, and cropping the stack. An AngleRefinement object can be passed in which can be used to translate the image correctly to the translation shifts found in an AngleRefinement.'''
    def __init__( self, res, dq, fpstack_fullres, angles, angleRef_obj=None ):
        self.res = res
        self.dq = dq
        self.fpstack_fullres = fpstack_fullres
        self.angles = angles
        self.n_angles = angles.n_angles

        self.translation_shifts_total = np.empty((fpstack_fullres.shape[0],2))

        self.get_freq_cutoff()
        self.make_res_stack()
        self.pstack = ifft2(self.fpstack, norm='ortho').real
        self.N_pstack = self.pstack.shape
        if angleRef_obj:
            self.angleRef = angleRef

    def get_freq_cutoff( self ):
        self.qres_ratio = 1 / (2 * self.res)    # Get resolution ratio in inverse coordinates (1/resolution)
        self.freqcutoff = np.ceil( self.qres_ratio / self.dq )  # Previously r! Determine cutoff frequencies for each resolution in pixels   
    
    def make_res_stack( self ):
        ''' Using our full-resolution fourier-space stack (N_fullres), our full-resolution nyquist-cutoff fourier-space stack (fpstack_fullres), and a resolution ratio (res_ratio) make a fourier-space stack with resolution = res(fpstack_fullres) * res_ratio '''
        r0 = self.freqcutoff
        n_proj = self.fpstack_fullres.shape[0]
        N_res = np.array([4 * r0, 4 * r0]).astype(np.int)
        N_fullres = np.array(self.fpstack_fullres.shape[1:])
        resolution_mask = np.prod(
                np.abs(fourier_coordinates_2D(N_fullres, 1 / N_fullres, centered=True)) < 
                        N_res[:, None, None] // 2, axis=0).astype(np.bool)
        s = self.fpstack_fullres[:, resolution_mask].reshape((n_proj, *N_res))
        w = slepian_window(N_res, self.qres_ratio)
        s *= w
        self.fpstack = s 
        self.N = N_res
        self.resolution_mask = resolution_mask
        self.w = w

    def normalize_pstack( self, ref_stack, n_proj=None ):
        ''' Normalize each stack's intensity to stack_ind's intensity '''
        if not n_proj:
            n_proj = self.n_angles 
        t  = np.sum( ref_stack[ n_proj // 2 ])
        ts = np.sum( self.pstack[ n_proj // 2 ])
        self.pstack *= t / ts  # Updates res_stacks without re-assigning s to res_stacks
        print(f'stack {self.res} sum: {np.sum(self.pstack[n_proj // 2])}')
    
    def plot_stack( self , fig_path, res_ratio): # Broken, res/res_ratio mixup? fig_path is not specified.
        if pplot:
            plotmosaic(s, f'Tilt series overview {res}x downsampling', dpi=600,
                   savePath=f'{fig_path}tilt_series_upsampled{res_ratio / 2}x')
    def crop( self, crop, pf='pstack', original_pixels=False, keep_original=False ):
        ''' Crops either fpstack (fp='fpstack') or pstack (fp='pstack') by amount given in crop. Crop can be a array/list with pattern [crop], [cropx,cropy],
        or [cropy_start, cropy_end, cropx_start, cropx_end] where crop/cropx are the number of pixels to shave off of each side respctively. All crop values
        should be in pixels of the original input stack.'''
        if not isinstance(crop, list):
            crop = [crop]
        if len(crop) == 2:
            crop = [crop[0], crop[0], crop[1], crop[1]]
        elif len(crop) == 1:
            crop = [crop[0], crop[0], crop[0], crop[0]]
        if original_pixels:
            crop = [c // self.res for c in crop ]
        #print('Crop:{}'.format(crop))
        
        if pf == 'fpstack':
            if keep_original:
                self.original_fpstack = self.fpstack
            if crop[1] == 0:
                self.fpstack = self.fpstack[:, crop[0]:, :]
            else:
                self.fpstack = self.fpstack[:, crop[0]:-crop[1],:]
            if crop[3] == 0:
                self.fpstack = self.fpstack[:, :, crop[2]:]
            else:
                self.fpstack = self.fpstack[:, :, crop[2]:-crop[3]]
        elif pf == 'pstack':  
            if keep_original:
                self.original_pstack = self.pstack
            if crop[1] == 0:
                self.pstack = self.pstack[:, crop[0]:, :]
            else:
                self.pstack = self.pstack[:, crop[0]:-crop[1],:]
            if crop[3] == 0:
                self.pstack = self.pstack[:, :, crop[2]:]
            else:
                self.pstack = self.pstack[:, :, crop[2]:-crop[3]]
            #self.pstack  =  self.pstack[ :, crop[0]:-crop[1], crop[2]:-crop[3] ]    

        self.N  = self.fpstack.shape[1:]
        self.N_pstack = self.pstack.shape

    def shift( self, translation_shifts=None, translations_upsample_factor=None ):
        ''' Shift res.pstack by translation_shifts*translations_upsample_factor/self.res. If translation_shifts/translations_upsample_factor then the translation_shifts from the angleRef_object passed into the Res class is used.'''
        if translation_shifts is None:
            translation_shifts = self.angleRef.translation_shifts
        if translations_upsample_factor is None:
            translations_upsample_factor = self.angleRef.res
        upsample_factor = translations_upsample_factor/self.freqcutoff 
        self.translation_shifts = translation_shifts * upsample_factor
        for i in range(self.n_angles):
            self.pstack[i] = ifftn(fourier_shift(fftn(self.pstack[i]), self.translation_shifts[i]) ) # '''Should this be res_ind? or upsample[ind]/upsample[res_ind]?'''
        self.translation_shifts_total += self.translation_shifts

    def upsample_vol(vol, upsample_factor, abs=False):
        padding = [int((upsample_factor * vols - vols)/2) for vols in vol.shape]
        padding = list(zip(padding,padding))
        vol = ifftn(fftshift(np.pad(fftshift(fftn(vol)), padding)))
        if abs:
            vol = np.abs(vol)
        return vol

class AngleRefinement():

    def __init__( self, res_obj, opts=None, iterations=10, angle_trials=6, angle_trial_start_range=np.deg2rad(.3), registration_upsample_factor=10, fig_path='~/tomo_results/' ):
        self.fig_path = fig_path
        if not opts:
            # Define opts parameters
            opts = Param()
            opts.record_objective = True
            opts.verbose = False
            opts.string_header = '     '
            opts.max_iters = 200
            opts.tol = 1e-5
            opts.accelerate = False
            opts.adaptive = True
            opts.restart = True
            # step size, is picked automatgically if you leave this out. Sometimes the convegence curve looks weird,
            # and you have to set it manually
            opts.tau = 20
        self.opts = opts

        # Options/setup for the angle search
        self.angles = res_obj.angles
        self.n_angles = res_obj.n_angles
        self.refinement_iterations = iterations #10
        self.angle_trial_start_range = angle_trial_start_range #np.deg2rad(0.3)
        self.angle_trials = angle_trials #6

        # Options/variables concerning pstack, precision, upsampling, 
        self.pstack = res_obj.pstack
        self.N = res_obj.N
        self.N_pstack = res_obj.N_pstack
        self.registration_upsample_factor = registration_upsample_factor #10
        self.resolution_upsample_factor = res_obj.res
        self.translation_shifts = np.zeros((res_obj.angles.n_angles, 2))
        self.precision = self.registration_upsample_factor * self.resolution_upsample_factor

        # Set a padding amount for the alignment process
        N = self.N_pstack
        self.pad_width = ((N[0] // 4, N[0] // 4), (N[1] // 4, N[1] // 4))
       
        self.best_angles = self.angles.angles
        
        # Set up A, At, and y
        self.vol_shape = ( self.N_pstack[1], self.N_pstack[2], self.N_pstack[2])
        real_space_extent = np.array([self.N_pstack[1], self.N_pstack[1], self.N_pstack[2]])

        self.A, self.At = ray_transforms(real_space_extent, self.N_pstack[1:], angles.n_angles, interp='nearest')
        if pplot:
            plot(np.sum(self.pstack, 0), 'sum(stack res)')
    def refine_theta(self, pplot=False):
        self.refine( phi=False, psi=False, theta=True, pplot=pplot)

    def refine_phi_psi( self, pplot=False):
        self.refine( phi=True, psi=True, theta=False, pplot=pplot)

    def refine_uniform_search( self, phi=True, psi=True, theta=True, pplot=pplot):
        pass
    
    def refine_tilt_axis( self, max_angle=90, step_size=1 ):
        ''' A systematic search for the best phi and psi angles to determine best tilt axis. Max_angle and stepsize in degrees.'''
        print("Refining tilt axis.....")
        #if step_size < 1:
        #    angles = np.deg2rad(range(-max_angle/step, max_angle, step_size))

        angles = np.deg2rad(np.arange(-max_angle, max_angle, step_size))
        self.axis_ref_objectives = np.zeros(len(angles))
        
        # Get set of angles to try, do angle trials, pick the best angles and apply the shifts
        angle_amount = np.zeros((len(angles)*len(angles),3))
        angle_amount[:,0] = np.repeat(angles, len(angles))
        angle_amount[:,2] = np.tile(angles, len(angles))
        angle_trial_num = angle_amount.shape[0]
        
        self.tilt_axis_losses, self.tilt_axis_shifts, self.tilt_axis_trial_angles= self.do_angle_trials( self.best_angles, angle_trial_num, angle_amount, i_refine=0, phi=True, theta=False, psi=True, uniform_search=True)
        self.best_angles, shifts = self.pick_best_angles( self.tilt_axis_trial_angles, self.tilt_axis_shifts, self.tilt_axis_losses, same_angles_all_projections=True)
        self.apply_shifts_from_best_angles( shifts )
        if pplot:
            plot_rotating_point(self.best_angles, f'Best angles iteration {i_refine+1}', dpi=600,
                            savePath=f'{self.fig_path}best_angles')
            np.save(path + 'fasta.npy', sol.cpu().numpy())
        print("Done with Tilt-Axis refinement")
    
    def refine(self, phi=True, psi=True, theta=True, pplot=pplot ):

        if not self.mu_best:
            self.pick_regularizer( self.angles, opts, mus=Options.mus, pplot=pplot)
         
        self.refinement_objectives = np.zeros((self.refinement_iterations))
        #x0 = th.zeros(*vol_shape, dtype=th.float32).cuda()
        #sol, out, opts_out = fasta_tomography_nonnegative_shrink(self.A, self.At, x0, self.y, self.mu_best, best_angles, opts)
        
        # Set a padding amount for the alignment process
        N = self.N_pstack
        self.pad_width = ((N[0] // 4, N[0] // 4), (N[1] // 4, N[1] // 4))
        
        
        # Refine the translations and angular alignment from a downsampled, cropped version of the projections
        for i_refine in range(self.refinement_iterations):  
            print(f'it {i_refine+1:-4d}/{self.refinement_iterations:-4d}: reconstructing ...')
        
            # TODO: Not sure if x0 and y need to be redefined every time. Can't tell if they're being altered
            x0 = th.zeros(*self.vol_shape, dtype=th.float32).to(device)
            y = th.as_tensor(np.transpose(self.pstack, (1, 0, 2))).contiguous().float().to(device)
            sol, out, opts_out = fasta_tomography_nonnegative_shrink(self.A, self.At, x0, y, self.mu_best, self.best_angles, self.opts) # Solve the optimization problem
            self.refinement_objectives[i_refine] = out.objective[-1]                                                               # Record the objective after solving
            self.pstack, nil = self.refine_translation( sol, y, self.best_angles, i_refine )                                    # Refine the translation with this set of angles
      
            # Get set of angles to try, do angle trials, pick the best angles and apply the shifts
            angle_amount= (1 - (i_refine / self.refinement_iterations)) * self.angle_trial_start_range # Angle trial range: (3, 8/3, 7/3, 2, 5/3, 4/3, 1, 2/3, 1/3 ,0)
            self.trial_losses, self.trial_shifts, self.trial_angles = self.do_angle_trials( self.best_angles, self.angle_trials, angle_amount, i_refine, phi=phi, theta=theta, psi=psi) 
            self.best_angles, shifts = self.pick_best_angles( self.trial_angles, self.trial_shifts, self.trial_losses )
            self.apply_shifts_from_best_angles( shifts )
        if pplot:
            plot_rotating_point(self.best_angles, f'Best angles iteration {i_refine+1}', dpi=600,
                            savePath=f'{self.fig_path}best_angles')
            #np.save(path + 'fasta.npy', sol.cpu().numpy())
        print("Done with Angle refinement")
    
    
    def do_angle_trials(self, best_angles, angle_trials, angle_amount, i_refine, phi=True, theta=True, psi=True, uniform_search=False):
        ''' Do the angle refinement. '''
        print(f'it {i_refine+1:-4d}/{self.refinement_iterations:-4d}: reconstructing for angle refinement ...')
        n_angles = best_angles.shape[1]
        x0 = th.zeros(*self.vol_shape, dtype=th.float32).to(device)
        y = th.as_tensor(np.transpose(self.pstack, (1, 0, 2))).contiguous().float().to(device)
        sol, out, opts_out = fasta_tomography_nonnegative_shrink(self.A, self.At, x0, y, self.mu_best, best_angles, self.opts)     # Solve the optimization problem
        print(f'best R-factor: {out.R_factors[-1]}')
        if not uniform_search:    
            print(f'it {i_refine+1:-4d}/{self.refinement_iterations:-4d}: angle refinement with range {np.rad2deg(angle_amount):2.2f} deg ...')
        trial_losses = np.zeros((angle_trials + 1, n_angles))
        trial_angles = np.zeros((angle_trials + 1, *best_angles.shape))
        trial_shifts = np.zeros((angle_trials + 1, n_angles, 2))
        last = 0
        for t in range(angle_trials):
            if uniform_search:
                if angle_amount[t,0] != last:
                    last = angle_amount[t,0]
                    print('    Phi: {}'.format(np.rad2deg(angle_amount[t,0])))
            offsets = np.zeros(best_angles.shape)
            if phi:
                if uniform_search:
                    offsets[0,:] = np.ones((1, best_angles.shape[1]))*angle_amount[t,0]
                else:
                    offsets[0,:] = uniform(-angle_amount, angle_amount, (1,best_angles.shape[1]))
            if theta:
                if uniform_search:
                    offsets[1,:] = np.ones((1, best_angles.shape[1]))*angle_amount[t,1]
                else:
                    offsets[1,:] = uniform(-angle_amount, angle_amount, (1,best_angles.shape[1]))
            if psi:
                if uniform_search:
                    offsets[2,:] = np.ones((1, best_angles.shape[1]))*angle_amount[t,2]
                else:
                    offsets[2,:] = uniform(-angle_amount, angle_amount, (1,best_angles.shape[1]))
            
            trial_angles[t] = offsets + best_angles

            trial_projs, trial_shifts[ t, :,:] = self.refine_translation(sol, y, trial_angles[t], i_refine, pad_width=self.pad_width, trials=True)
            trial_projs= th.transpose(th.tensor(trial_projs, device=device), 1, 0)
            trial_losses[ t ]  = ( th.norm(trial_projs - y, dim=(0, 2)) ** 2).cpu().numpy() #
    
        # Do this one more time without the translation refinement and record 
        trial_angles[ -1 ] = best_angles   # Maybe this goes outside the t loop?
        trial_proj         = th.zeros_like(y)
        trial_proj         = self.A(sol, out=trial_proj, angles=trial_angles[t])
        trial_losses[-1]   = (th.norm(trial_proj - y, dim=(0, 2)) ** 2).cpu().numpy()
        return trial_losses, trial_shifts, trial_angles
        
    def refine_translation( self, model, y, angles, i_refine, pad_width=None, trials=False, pplot=False ):
        '''Register projection with subpixel precision. Default is to record the translations found (trials=False). If trials=True, translations will not be recorded within the class because we are just testing out different angles and will record the translations externally until the "best" angle combo is found.'''
        #print(f'it {i_refine+1:-4d}/{self.refinement_iterations:-4d}: subpixel registration ...')
        #print('Trial Angles: {}'.format(angles))
        n_angles               = angles.shape[1]
        rms_translation_errors = np.zeros((n_angles), dtype=np.float32)
        #translation_shifts     = np.zeros((n_angles, 2))
        current_shifts         = np.zeros((n_angles, 2))
        pstack                 = y.cpu().numpy()
        
        # Check to make sure pstack has the right shape
        if pstack.shape[0] == n_angles:
            pass
        elif pstack.shape[1] == n_angles:
            pstack = np.transpose(pstack, (1, 0, 2))
        elif pstack.shape[2] == n_angles:
            pstack = np.transpose(pstack, (2, 1, 0))
    
        # Calculate projections from our estimate, put into correct dimensions
        y_model = th.zeros_like(y)
        y_model = self.A(model, out=y_model, angles=angles)
        if y_model.shape[1] == n_angles:
            y_model = th.transpose(y_model, 0, 1).cpu().numpy()
        elif y_model.shape[2] == n_angles:
            y_model = th.transpose(y_model, 0, 2).cpu().numpy()           # Transpose the projections, and put onto the cpu
    
        for i in range(n_angles):
            proj   = self.pad( pstack[i,:,:], pad_width )
            y_proj = self.pad( y_model[i, :, :], pad_width )
            shift, rms_translation_errors[i], diffphase = register_translation( y_proj, proj, self.precision ) # Compare our stack of experimental projections to the projections produced by our current model,
        
            #if isinstance( self.trial_shifts, np.ndarray): 
            #    self.trial_shifts[ i_refine, i] = shift  #TODO: Figure out what this is and what iter does
            if not trials:
                self.translation_shifts[i] += shift    # Keep track of the total shifts made within the angle_refinement class (only if we aren't trying out a whole bunch of different angles to find the best
            current_shifts[i] = shift              # Output current_shifts 
        
            proj = ifftn(fourier_shift(fftn(proj), shift))# Translate the image by the amount found by resgister_translation (shifts)
            pstack[i] = self.unpad( proj, pad_width) #th.as_tensor( self.unpad( proj, pad_width ) ).to(device) # #
    
        return pstack, current_shifts
 

    def pick_best_angles( self, trial_angles, trial_shifts, trial_losses, same_angles_all_projections=False ):
        ''' After you've tried all the different angles, pick the best for each projection, as well as the best shifts for that particular angle choice '''
        if same_angles_all_projections:
            min_ind = np.argmin(np.mean(trial_losses, axis=1))
            best_angles = np.squeeze(trial_angles[min_ind,:,:])
            shifts = np.squeeze(trial_shifts[min_ind,:,:])
        else:
            min_loss = np.min( trial_losses, axis=0)     # Minimum loss across angle trials
            min_ind  = np.argmin( trial_losses, axis=0)  # Index of minimum loss from the angle trials
            #print("min losses: ",[float("%.3f"%min) for min in min_loss])
            best_angles = np.squeeze(
            np.stack([a[min_ind[i]] for i, a in enumerate(np.split(trial_angles, np.arange(1, self.n_angles), axis=2))])).T
            shifts = np.squeeze(np.stack([a[min_ind[i]] for i, a in enumerate(np.split(trial_shifts, np.arange(1, self.n_angles), axis=1))]))
        print("best angles: ",[float("%.3f"%rd) for rd in np.rad2deg(best_angles[0])])
        print("             ",[float("%.3f"%rd) for rd in np.rad2deg(best_angles[1])])
        print("             ",[float("%.3f"%rd) for rd in np.rad2deg(best_angles[2])])

        return best_angles, shifts
    
    def apply_shifts_from_best_angles(self, shifts):
        ''' Apply shifts to self.pstack and record the added shifts in self.translation_shifts'''
        if not self.pad_width:
            pup    = np.ceil(np.abs(np.max(shifts[1]))).astype(int)
            pdown  = np.ceil(np.abs(np.min(shifts[1]))).astype(int)
            pright = np.ceil(np.abs(np.max(shifts[0]))).astype(int)
            pleft  = np.ceil(np.abs(np.min(shifts[0]))).astype(int)
            self.pad_width = [[pup,pdown], [pleft,pright]]
        for i in range(self.n_angles):
            s = np.pad(self.pstack[i], self.pad_width)
            s = ifftn(fourier_shift(fftn(s), -shifts[i]))
            self.pstack[i,:,:] = s[self.pad_width[0][0]:-self.pad_width[0][1], self.pad_width[1][0]:-self.pad_width[1][1]]
            self.translation_shifts[i] += -shifts[i]
    def pick_regularizer( self, angles, opts, pplot=False, mus=np.array([1e2, 5e1, 1e1, 5e0, 1e0, 5e-1, 1e-1, 5e-2, 1e-1, 5e-3, 1e-3])):
        '''Do angle refinement on low-resolution data also look for best translations while at different angles'''
    
        # Check for best regularization parameter
        mu_objectives = np.zeros_like(mus)
        mu_g_values = np.zeros_like(mus)
        mu_solution_norm = np.zeros_like(mus) 
        
        x0 = th.zeros(*self.vol_shape, dtype=th.float32).to(device)
        y = th.as_tensor(np.transpose(self.pstack, (1, 0, 2))).contiguous().float().cuda()
        for i, mu in enumerate(mus):
            sol, out, opts = fasta_tomography_nonnegative_shrink(self.A, self.At, x0, y, mu, angles.angles, opts)
            mu_objectives[i] = out.objective[-1]
            mu_g_values[i] = out.g_values[-1]
            mu_solution_norm[i] = th.norm(sol)
    
        if pplot:
            font = {'family': 'serif',
                    'color': 'black',
                    'weight': 'normal',
                    'size': 6,
                    }
            f, a = plt.subplots(1, 2, dpi=300)
            a[0].loglog(mu_g_values, mu_objectives)
            a[0].set_ylabel('regularizer loss')
            a[0].set_xlabel('objective loss')
            for g, o, mu in zip(mu_g_values, mu_objectives, mus):
                a[0].text(g, o, f'mu = {mu:2.2g}', fontdict=font)
            a[1].loglog(mu_objectives, mu_solution_norm)
            a[1].set_ylabel('solution norm')
            a[1].set_xlabel('objective loss')
            for g, o, mu in zip(mu_objectives, mu_solution_norm, mus):
                a[1].text(g, o, f'mu = {mu:2.2g}', fontdict=font)
            plt.show()
        mu_best = "Some function which takes information about how the reconstruction performed and determines the best mu."
        self.mu_best = mu_best
    def pad( self, proj, pad_width ): 
        if pad_width:
            proj = np.pad(proj, pad_width)
        else:
            proj = proj
        return proj

    def unpad( self, proj, pad_width ):
        if pad_width:
            proj = proj[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]
        return proj
"""
def refine_alignment(stack_res, vol_shape, refinement_iterations, precision, translation_shifts, A, At, mu, best_angles, opts):
    # Translation Refinement
    for i_refine in range(refinement_iterations):
        x0 = th.zeros(*vol_shape, dtype=th.float32).to(device)
        y = th.as_tensor(np.transpose(stack_res, (1, 0, 2))).contiguous().float().to(device)
        print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing ...')
        sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, best_angles, opts) # Solve the optimization problem
        refinement_objectives[i_refine] = out.objective[-1]                       # Record the objective after solving
        # register projection with subpixel precision
        print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: subpixel registration ...')
        y_model = th.zeros_like(y)
        y_model = A(sol, out=y_model, angles=best_angles)                         # Calculate projections from our object estimate
        y_model = th.transpose(y_model, 0, 1).cpu().numpy()                                           # Transpose the projections, and put onto the cpu

        stack_res, translation_shifts       = calculate_and_shift(angles, y_model, stack_res,  precision, i_refine=i_refine, translation_shifts=translation_shifts, plot=pplot)    # Calculate the translation shifts necessary for the projections and shift
        min_ind, trial_angles, trial_shifts = refine_angle_alignment(angle_trial_start_range, angle_trials, A, At, x0, y, mu, best_angles, opts, i_refine=i_refine)
        best_angles, shifts                 = pick_best_angles(min_ind, trial_angles, angles, trial_shifts, i_refine=i_refine)
        stack_res                       = apply_shifts_from_best_angles(shifts, stack_res, pad_width)
        translation_shifts                  = add_shifts(shifts, translation_shifts, angles)
    return best_angles, translation_shifts
def refine_angle_alignment(angle_trial_start_range, angle_trials, A, At, x0, y, mu, best_angles, opts, i_refine=i_refine):
    ''' Do the angle refinement. (Call inside of the refinement_alignment) '''
    print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing for angle refinement ...')
    sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, best_angles, opts)     # Solve the optimization problem
    print(f'best R-factor: {out.R_factors[-1]}')

    # Compute trial angle losses and find best
    angle_trial_range = (1 - (i_refine / refinement_iterations)) * angle_trial_start_range # Angle trial range: (3, 8/3, 7/3, 2, 5/3, 4/3, 1, 2/3, 1/3 ,0)
    print(
        f'it {i_refine:-4d}/{refinement_iterations:-4d}: angle refinement with range {np.rad2deg(angle_trial_range):2.2f} deg ...')
    trial_losses = np.zeros((angle_trials + 1, angles.n_angles))
    trial_angles = np.zeros((angle_trials + 1, *best_angles.shape))
    trial_shifts = np.zeros((angle_trials + 1, angles.n_angles, 2))
    for t in range(angle_trials):
        random_offsets = uniform(-angle_trial_range, angle_trial_range, best_angles.shape)
        # random_offsets[0] = 0
        # random_offsets[2] = 0
        trial_angles[t] = random_offsets + best_angles
        trial_proj = th.zeros_like(y)
        trial_proj = A(sol, out=trial_proj, angles=trial_angles[t])
        trial_proj = th.transpose(trial_proj, 0, 1)
        '''y = th.transpose(y, 0, 1)'''
        tt, trial_shifts = calculate_and_shift(angles, y, trial_proj, precision, i_refine=i_refine, iter=t, trial_shifts=trial_shifts, pad_width=pad_width, translation_shifts=False, plot=pplot)
        trial_proj= th.transpose(tt, 1, 0)
        '''trial_proj[:, n, :] = th.as_tensor(tt[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]).to(device)'''
        #for n in range(angles.n_angles):
        #   yn = np.pad(y[:, n, :].cpu().numpy(), pad_width)
        #   y_modeln = np.pad(trial_proj[:, n, :].cpu().numpy(), pad_width)
        #   shift, _, diffphase = register_translation(yn, y_modeln, precision)
        #   trial_shifts[t, n] = shift
        #   tt = ifftn(fourier_shift(fftn(y_modeln), shift)).real
        #   trial_proj[:, n, :] = th.as_tensor(
        #       tt[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]).to(device)

        trial_losses[t] = (th.norm(trial_proj - y, dim=(0, 2)) ** 2).cpu().numpy() #
        trial_angles[-1] = best_angles

    trial_proj = th.zeros_like(y)
    trial_proj = A(sol, out=trial_proj, angles=trial_angles[t])
    trial_losses[-1] = (th.norm(trial_proj - y, dim=(0, 2)) ** 2).cpu().numpy()
    min_loss = np.min(trial_losses, axis=0) # Minimum loss across angle trials
    mini_ind = np.argmin(trial_losses, axis=0) # Index of minimum loss from the angle trials
    print(f'min indices: {mini_ind}')
    return mini_ind, trial_angles, trial_shifts
def pick_best_angles( min_ind, trial_angles, angles, trial_shifts, i_refine=i_refine, plot=pplot):
    ''' After you've tried all the different angles, pick the best for each projection, as well as the best shifts for that particular angle choice '''
    # pick angles with minimum error
    best_angles = np.squeeze(
        np.stack([a[min_ind[i]] for i, a in enumerate(np.split(trial_angles, np.arange(1, angles.n_angles), axis=2))])).T
    shifts = np.squeeze(
        np.stack([a[min_ind[i]] for i, a in enumerate(np.split(trial_shifts, np.arange(1, angles.n_angles), axis=1))]))
    print(f'best angles: {np.rad2deg(best_angles[1])}')
    if plot:
        plot_rotating_point(best_angles, f'Best angles iteration {i_refine}', dpi=600,
                            savePath=f'{fig_path}best_angles')
        np.save(path + 'fasta.npy', sol.cpu().numpy())

    return best_angles, shifts
def apply_shifts_from_best_angles(shifts, stack_res, pad_width):
    for i in range(angles.n_angles):
        s = np.pad(stack_res[i], pad_width)
        s = ifftn(fourier_shift(fftn(s), -shifts[i]))
        stack_res[i] = s[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]
    return stack_res
def add_shifts(shifts, translation_shifts, angles):
    for i in range(angles.n_angles):
        translation_shifts[i] += -shifts[i]
    return translation_shifts

# Find a good mu
#mus = np.array([1e2, 5e1, 1e1, 5e0, 1e0, 5e-1, 1e-1, 5e-2, 1e-1, 5e-3, 1e-3])
#mu = pick_regularizer(A, At, x0, y, angles, opts, pplot=pplot, mus=mus)

    def plot_angles( self, sol ):
    rec = sol.cpu().numpy()
    f, a = plt.subplots(1, 3, dpi=300)
    a[0].imshow(rec[vol_shape[0] // 2, :, :])
    a[0].set_xlabel(f'z')
    a[0].set_ylabel(f'x')
    a[1].imshow(rec[:, vol_shape[0] // 2, :])
    a[1].set_xlabel(f'z')
    a[1].set_ylabel(f'y')
    a[2].imshow(rec[:, :, vol_shape[0] // 2])
    a[2].set_xlabel(f'x')
    a[2].set_ylabel(f'y')
    plt.show()
    f.savefig(f'{self.fig_path}xyz_view.png', dpi=600)
    # %%
    f, a = plt.subplots(1, 3, dpi=300)
    a[0].semilogy(out.residuals)
    a[0].set_title(r"$||x_n-x_{n+1}||^2$")
    a[1].semilogy(out.objective)
    a[1].set_title(r"$||y-Ax||^2$")
    a[2].semilogy(out.R_factors)
    a[2].set_title(r"R-factor $=\frac{||y-Ax||_1}{||y||_1}$")
    plt.show()
    print(f'best R-factor: {out.R_factors[-1]}')
    # %%
    plot_rotating_point(best_angles, f'Best angles iteration {i_refine}', savePath=f'{self.fig_path}best_angles.png',
                        dpi=300)
    plot_rotating_point(angles, f'Start angles iteration {i_refine}', savePath=f'{self.fig_path}start_angles.png', dpi=300)


# Get full, upsampled resolution stack, apply translation shifts in k-space,
ind = -1
stack_res2 = res_stacks[ind].copy()
for i in range(angles.n_angles):
    stack_res2[i] = ifftn(fourier_shift(fftn(stack_res2[i]), translation_shifts[i] * upsample_factors[res_ind])) # '''Should this be res_ind? or upsample[ind]/upsample[res_ind]?'''
if pplot:
    plotmosaic(stack_res2, dpi=600, savePath=f'{fig_path}stack_aligned_fullres.png')
# %%
m = 160
stack_res3 = stack_res2[:, m:-m, m:-m]
#pplot=True
if pplot:
    plot(np.sum(stack_res3, 0), 'sum(stack res)')
# %%
# stack_res3 = stack_res

ps = stack_res3.shape
projection_shape = ps[1:]
vol_shape = (projection_shape[0], projection_shape[1], projection_shape[1])
real_space_extent = np.array([projection_shape[0], projection_shape[0], projection_shape[1]])
#support = upsample_vol(support, upsample_factors[ind], abs=True)
#support = th.as_tensor(support[m:-m,m:-m,m:-m], dtype=th.bool).contiguous().to(device)
support = th.as_tensor(support,dtype=th.bool).no_grad().contiguous().cpu() #.to(device)

opts = Param()
opts.record_objective = False
opts.verbose = True
opts.string_header = '     '
opts.max_iters = 200
opts.tol = 1e-5
opts.accelerate = False
opts.adaptive = False
opts.restart = True
opts.tau = 20
mu = 100e-2

A, At = ray_transforms(real_space_extent, projection_shape, angles.n_angles, interp='linear')
x0 = th.zeros(*vol_shape, dtype=th.float32).to(device)
y = th.as_tensor(np.transpose(stack_res3, (1, 0, 2))).contiguous().float().to(device)
print("reconstructing...")#f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing ...')
del sol
th.cuda.empty_cache()
sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, best_angles, opts)
#sol, out, opts_out = fasta_tomography_nonnegative_shrink_support_constraint(A, At, x0, y, mu, best_angles, support, device, opts)
# %%
mp = th.zeros_like(y)
mp = A(sol, out=mp, angles=best_angles)

if pplot:
    i = np.ceil(stack_res3.shape[1] / 2, dtype=int)
    plot(np.hstack([y[:, i, :].cpu(), mp[:, i, :].cpu()]), 'y           y_model')
# %%
diff = np.transpose((y - mp).cpu().numpy(), (1, 0, 2))
if pplot:
    plotmosaic(diff, 'y - y_model', dpi=900)
# %%
rec = sol.cpu().numpy()
if pplot:
    f, a = plt.subplots(1, 3, dpi=300)
    a[0].imshow(rec[vol_shape[0] // 2, :, :])
    a[0].set_xlabel(f'z')
    a[0].set_ylabel(f'x')
    a[1].imshow(rec[:, vol_shape[0] // 2, :])
    a[1].set_xlabel(f'z')
    a[1].set_ylabel(f'y')
    a[2].imshow(rec[:, :, vol_shape[0] // 2])
    a[2].set_xlabel(f'x')
    a[2].set_ylabel(f'y')
    plt.show()
    f.savefig(f'{fig_path}xyz_fullres.png', dpi=300)
    # %%
    f, a = plt.subplots(1, 3, dpi=300)
    a[0].semilogy(out.residuals)
    a[0].set_title(r"$||x_n-x_{n+1}||^2$")
    a[1].semilogy(out.objective)
    a[1].set_title(r"$||y-Ax||^2$")
    a[2].semilogy(out.R_factors)
    a[2].set_title(r"R-factor $=\frac{||y-Ax||_1}{||y||_1}$")
    plt.show()
    f.savefig(f'{fig_path}residuals_objective__fullres.png', dpi=300)
print(f'best R-factor: {out.R_factors[-1]}')

np.save(pathout + 'fasta_best3.npy', sol.cpu().numpy())"""
if '__name__' == '__main__':
    if not os.path.exists(pathout):
        os.mkdir(pathout)
    recon = Recon( projections, angles, dx, pathout,fig_path, resolution_cutoff, recon_resolution, recon_crop, opts_r )
    recon.do_a_reconstruction( angle_refine=2)
