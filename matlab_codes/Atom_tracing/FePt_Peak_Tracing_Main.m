
%%
path = '/home/philipp/drop/Public/'
fn = 'fasta_best3.npy'

v1 = readNPY([path fn]);
v = v1(:,:,:);
imagesc(squeeze(v(:,224,:)))
%% Init some variables
Dsetvol = v1;
% Sdn_pad = zeros(448,60,448);
% Sdn_pad(:,5:54,:) = Dsetvol;
Sdenoise_all_2 = Dsetvol;
Th = 50;
% Pixel Resolution in Angstroem
Res = 0.178;
% Sdenoise_all_2 = Dsetvol;

BoxSize0=3; %box size used for average when sorting peaks
BoxSize1=3; %box size used to find maxima
BoxSize2=11; %box size used box for fitting off-center gauss
BoxSize3=12; %used to compute visualization matrix

[Xsize, Ysize, Zsize] = size(Sdenoise_all_2);  

%% Trace peaks
    
MaxNumberPeaks=50000; %maximum number of peaks to find


% minimum distance constraint in the unit of pixel
Dmin = 2/Res; % this corresponds to 2 Angstrom for FePt reconstruction

    
DataMatrix = Sdenoise_all_2;
%%
%find possible atoms (minimal distance contraint applied)
% atom_pos is the xyz positions (in the unit of pixel) of found local maxima peaks
% close_pos is the xyz positions of peaks which violate the minimum distance constraint
% statsI is the initial Gaussian fitting parameters of each peak
% stats is the final Gaussian fitting parameters of each peak
[atom_pos, close_pos,  statsI, stats, CurrData ] = find_possible_atoms(DataMatrix, Dmin, MaxNumberPeaks, BoxSize0, BoxSize1,BoxSize2,Th);
%%
DataName = 'Pd5fold'
ourputstring = '1'
TracingStep = 1
%save result
eval(['save ' DataName '_' ourputstring '_step' num2str(TracingStep)  '.mat atom_pos close_pos Dmin stats statsI DataMatrix MaxNumberPeaks'])
%%
r = atom_pos;
figure(2)
scatter3(r(3,:),r(1,:),r(2,:),5, ...        
        'MarkerFaceColor',[0 .75 .75])
 %%
 r = permute(atom_pos, [2 1]);
 r = r(1:16147,:);
 %%
radiusNN         = 4/0.174;   % in pixels / voxels
minNumNNsAllowed = 22;  % if site has less NNs than this, it will be removed
maxNNbins        = 100;  % histogram
flagLocalIntensityCriteria = false;
% low local intensity ratio
distCheck        = 8;
numNNcheck       = 16;
minIntRatio      = 0.25;
removeSitesAtomTraceHelper(r, size(DataMatrix), flagLocalIntensityCriteria, radiusNN, ...
    minNumNNsAllowed, maxNNbins, distCheck, numNNcheck, minIntRatio);
%%
r1 = removeSitesAtomTraceHelper(r, size(DataMatrix), flagLocalIntensityCriteria, radiusNN, ...
    minNumNNsAllowed, maxNNbins, distCheck, numNNcheck, minIntRatio);
%%
figure(2)
scatter3(r1(:,3),r1(:,1),r1(:,2),5, ...        
        'MarkerFaceColor',[0 .75 .75])