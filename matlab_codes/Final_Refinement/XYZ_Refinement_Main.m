% Author: Rui Xu and Yongsoo Yang
% email: yongsoo.ysyang@gmail.com (YY)
% Jianwei (John) Miao Coherent Imaging Group
% University of California, Los Angeles



function XYZ_Refinement_Main()

%parpool;

% import Euler angles
angles = importdata('FePt_final_Angles.mat');

% import atomic coordinates
% NOTE: this atomic corrdinates are used to calculate the F_calc to compare
% with the F_obs (calculated from measured projections). Therefore, these
% corrdinates should be in the original reconstruction orientation, not the
% rotated orientation for making {100} axes along the MATLAB array
% directions. The roration of coordinate with {100} convention to original
% convention can be done by simply applying a 3D rotation matrix to the 
% coordinates. For details about the rotation, please see the Rotation.zip
% file inside the GENFIRE package provided in the Coherent Imaging Group
% website.
% Also, the coordinates should be in Angstrom in current 
% implemenataion, with the "Center Pixel" of the reconstruction be at the
% Origin (i.e., x = y = z = 0). In MATLAB convention, "Center Pixel" of
% an array with size N is at N/2+1 if N is even, and at (N+1)/2 if N is
% odd.
model = importdata('current_FePt_model_coordinates.mat');

% import atom type (1=Fe, 2=Pt)
atoms = importdata('current_FePt_model_atomtypes.mat');

% import measured projections
projetions = importdata('FePt_Denoised_Projections.mat');

% crop projections to make odd-numbered array
projetions = projetions(2:256,2:256,:);

% initialize variables
% N: size of input projection
% Hf: (N-1)/2
% Nc: (N+1)/2
% M: # of projections
N = 255; Hf = 127; Nc = 128; M = 68;

% set pixel resolution in angstrom
res = 0.3725;

% resolution in Real space and Fourier space
delta_r = res; delta_k = 1/(N*delta_r);

% get atomic coordinates in Angstrom
pos = model * res;

% get each Euler angles
phis = angles(1,:);
thetas = angles(2,:);
psis = angles(3,:);

% calculate rotation matrices and normal vectors from the Euler angles
for hh = 1:M

vector1 = [0 0 1];
rotmat1 = MatrixQuaternionRot(vector1,phis(hh));
vector2 = [0 1 0];
rotmat2 = MatrixQuaternionRot(vector2,thetas(hh));
vector3 = [0 0 1];
rotmat3 = MatrixQuaternionRot(vector3,psis(hh));
rotmats(:,:,hh) =  rotmat3*rotmat2*rotmat1;

end

% initialize variables
Fobs = zeros(1,M*N*N);
Fcalc = zeros(1,M*N*N);
kx = zeros(1,M*N*N);
ky = zeros(1,M*N*N);
kz = zeros(1,M*N*N);
mask3D = zeros(1,M*N*N);

% run FFT to calculate F_obs from each projection
for nj = 1:M;

[kx0, kz0] = meshgrid(-Hf:1:Hf, -Hf:1:Hf);
ky0 = kx0; ky0(:) = 0;
k_plane0 = [kx0(:) ky0(:) kz0(:)]';
k_plane0 = k_plane0*delta_k;
k_plane = rotmats(:,:,nj)*k_plane0;
kx(1,1+(nj-1)*N*N:nj*N*N) = k_plane(1,:);
ky(1,1+(nj-1)*N*N:nj*N*N) = k_plane(2,:);
kz(1,1+(nj-1)*N*N:nj*N*N) = k_plane(3,:);
proj = projetions(:,:,nj);
Fobs_t = My_FFTN(proj);
Fobs(1,1+(nj-1)*N*N:nj*N*N) = reshape(Fobs_t,1,N*N);

end

% load Fourier mask and apply mask (resolution sphere binary mask)
load mask3D.mat;
ind = find(mask3D==0);
kx(ind) = [];
ky(ind) = [];
kz(ind) = [];
Fobs(ind) = [];
Fcalc(ind) = [];
L = numel(kx);

% k magnitude
q2 = kx.^2 + ky.^2 + kz.^2;

% obtain tabulated electron scattering form factor for Fe(Z=26) and
% Pt(Z=78)
% (from Kirkland, E. J. Advanced Computing in Electron Microscopy 2nd edn 
% Springer Science & Business Media, 2010).
fa78 = fatom_vector( sqrt(q2),78 );
fa26 = fatom_vector( sqrt(q2),26 );

ka.kx = kx; ka.ky = ky; ka.kz = kz;
ra.rx = pos(1,:); ra.ry = pos(2,:); ra.rz = pos(3,:);
fa(:,:) = [fa78; fa26; 0.5*(fa78+fa26)];

% pre-optimized B_A_n factors and H_A_n factors
bf = pos(1,:); bf(atoms==1) = 5.4855; bf(atoms==2) = 5.0360;
ht = pos(1,:); ht(atoms==1) = 0.5624; ht(atoms==2) = 1.1842;

fa_3 = fa(3,:);

% scale parameter to adjust atomic position shift stepsize for the fitting
alpha = 1;

% scale parameter to adjust the contribution from potential gradient and
% measure Foruer data error gradient.
lambda = 1.4015e+009;

for kk = 1:150

    kk

parfor hh = 1:L
Fcalc(hh) = fa_3(hh)*sum( ht.*exp( -2*pi*i*(ka.kx(hh)*ra.rx+ka.ky(hh)*ra.ry+ka.kz(hh)*ra.rz)-bf*q2(hh) ) );
end
k = kfactor(Fobs,Fcalc);
Fcalc = Fcalc*k;
err1 = sum( abs( abs(Fobs(:))-abs(Fcalc(:)) ) )/sum( abs(Fobs(:)) )
err2 = sum( abs( Fobs(:)-Fcalc(:) ).^2 );
pot = EamPot_FePt2(pos,atoms);
err3 = err2 + lambda*pot;

err1_arr(kk) = err1;
err2_arr(kk) = err2;
pot_arr(kk)  = pot;
err3_arr(kk) = err3;
pos_arr(:,:,kk) = pos;

% obtain the gradient vector for atomic potential and error for measured Fourier data from current configuration
[sca dra] = Grad_Dr3_Par(Fobs,Fcalc,ra,ka,fa,bf,ht);
[scb drb] = EamGrad_FePt2(pos,atoms);

% scale the gradient with scale aprameter
dra.drx = sca*dra.drx + lambda*scb*drb.drx; 
dra.dry = sca*dra.dry + lambda*scb*drb.dry;
dra.drz = sca*dra.drz + lambda*scb*drb.drz;
scale = sqrt( sum( (dra.drx).^2 + (dra.dry).^2 + (dra.drz).^2 ) );
dra.drx = dra.drx/scale; dra.dry = dra.dry/scale; dra.drz = dra.drz/scale;

% apply the shift vector for next iteration
ra.rx = ra.rx - dra.drx*alpha;
ra.ry = ra.ry - dra.dry*alpha;
ra.rz = ra.rz - dra.drz*alpha;

pos = [ra.rx; ra.ry; ra.rz];

% save the fitting iteration result obtained so far
save err1_arr_new.mat err1_arr;
save err2_arr_new.mat err2_arr;
save pos_arr_new.mat pos_arr;
save pot_arr_new.mat pot_arr;
save err3_arr_new.mat err3_arr;

end

%poolobj = gcp('nocreate');
%delete(poolobj);

end