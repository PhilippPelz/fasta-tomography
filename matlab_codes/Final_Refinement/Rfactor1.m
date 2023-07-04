
angles = importdata('FePt_ov4_test_M_obo_FFT_nopsi_new_angles_9.mat');
atoms = importdata('atoms.mat');
projetions = importdata('FePt_20131218_Li_projections_M.mat');
projetions = projetions(2:256,2:256,:);

N = 255; Hf = 127; Nc = 128; M = 68;
sc = 0.97; res = 0.3725*sc;
delta_r = res; delta_k = 1/(N*delta_r);

load pos_arr.mat;
pos = pos_arr(:,:,50);

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

for nj = 1:M;
    
    nj
[kx0, kz0] = meshgrid(-Hf:1:Hf, -Hf:1:Hf);
ky0 = kx0; ky0(:) = 0;
k_plane0 = [kx0(:) ky0(:) kz0(:)]';
k_plane0 = k_plane0*delta_k;
k_plane = rotmats(:,:,nj)*k_plane0;
kx = k_plane(1,:);
ky = k_plane(2,:);
kz = k_plane(3,:);
PJobs = projetions(:,:,nj);

L = numel(kx);
q2 = kx.^2 + ky.^2 + kz.^2;
fa78 = fatom_vector( sqrt(q2),78 );
fa26 = fatom_vector( sqrt(q2),26 );
ka.kx = kx; ka.ky = ky; ka.kz = kz;
ra.rx = pos(1,:); ra.ry = pos(2,:); ra.rz = pos(3,:);
fa(:,:) = [fa78; fa26; 0.5*(fa78+fa26)];
bf = pos(1,:); bf(atoms==1) = 5.4868; bf(atoms==2) = 4.8679;
ht = pos(1,:); ht(atoms==1) = 0.5814; ht(atoms==2) = 1.1688;
fa_3 = fa(3,:);
for hh = 1:L
Fcalc(hh) = fa_3(hh)*sum( ht.*exp( -2*pi*i*(ka.kx(hh)*ra.rx+ka.ky(hh)*ra.ry+ka.kz(hh)*ra.rz)-bf*q2(hh) ) );
end
Fcalc = reshape(Fcalc,N,N);
PJcalc = real(My_IFFTN(Fcalc));
k = kfactor(PJobs,PJcalc);
PJcalc = PJcalc*k;
rf = sum( abs(PJcalc(:)-PJobs(:)) )/sum( abs(PJobs(:)) )
rf_arr(nj) = err;

end

