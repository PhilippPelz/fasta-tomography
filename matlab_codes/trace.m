path = '/home/philipp/drop/Public/'
fn = 'fasta_best3.npy'

v1 = readNPY([path fn]);
sv = size(v1);
v = zeros(sv(1), 70, sv(2));
v(:,11:60,:) = v1(:,201:250,:);

imagesc(squeeze(v(:,11,:)))
%%
minimumIntensity = 0; 
sPeaks = initializeAtomTraceHelper(v, minimumIntensity);
r = sPeaks.peaksCand(:,1:3);

s = 50;
m = 8;

sub = (r(:,2) > s + m) | (r(:,2) < s - m);
r(sub, :) = [];
figure(1)
hold on;
imagesc(squeeze(v(:,s,:)))
scatter(r(:,3),r(:,1),'MarkerEdgeColor',[0 .1 .5],...
              'LineWidth',1.5)
%%
sPeaks = refineSitesAtomTraceHelper(v,sPeaks);
%%
r = sPeaks.peaksRefine(:,1:3);
s = 45;
m = 10;

sub = (r(:,2) > s + m) | (r(:,2) < s - m);
r(sub, :) = [];
figure(1)
hold on;
imagesc(squeeze(v(:,s,:)))
scatter(r(:,3),r(:,1),'MarkerEdgeColor',[0 .1 .5],...
              'LineWidth',1.5)
figure(2)
histogram(sPeaks.peaksRefine(:,4),100)
%%
sPeaks2 = sPeaks;
%%
% minIntensityPeak    = 50;
% loopsRefineAfterAdd = 2;
% 
% % Main loop
% for a0 = 1:numLoops
%     % Add new peaks
%     sPeaks2 = addSitesAtomTraceHelper(vol,sPeaks);
%     % Refine
%     for a1 = 1:loopsRefineAfterAdd
%         sPeaks2 = refineSitesSubtractNeighborAtomTraceHelper(vol,sPeaks2);
%     end
%     
%     % Merge peaks
%     sPeaks2 = mergeAndThresholdAtomTraceHelper(sPeaks2);
%     % Refine
%     sPeaks2 = refineSitesSubtractNeighborAtomTraceHelper(vol,sPeaks2);
%     
%     % Delete peaks
%     del = sPeaks2.peaksRefine(:,4) ...
%         + sPeaks2.peaksRefine(:,6) ...
%         < minIntensityPeak;
%     sPeaks2.peaksRefine(del,:) = [];
%     sPeaks2 = removeSitesAtomTraceHelper(sPeaks2);
%     % Refine
%     sPeaks2 = refineSitesSubtractNeighborAtomTraceHelper(vol,sPeaks2);
%     
%     Np = size(sPeaks2.peaksRefine,1);
%     disp(['        iter # ' num2str(a0) '/' num2str(numLoops) ... 
%         ' done, total number of sites = ' num2str(Np)])
% end
%%
sPeaks2 = addSitesAtomTraceHelper(v,sPeaks2);
%%
sPeaks2 = refineSitesSubtractNeighborAtomTraceHelper(v,sPeaks2);
%%
radiusNN         = 3/0.174;   % in pixels / voxels
minNumNNsAllowed = 11;  % if site has less NNs than this, it will be removed
maxNNbins        = 100;  % histogram
flagLocalIntensityCriteria = false;
% low local intensity ratio
distCheck        = 8;
numNNcheck       = 16;
minIntRatio      = 0.25;
removeSitesAtomTraceHelper(sPeaks2, flagLocalIntensityCriteria, radiusNN, ...
    minNumNNsAllowed, maxNNbins, distCheck, numNNcheck, minIntRatio);
%%
r = sPeaks2.peaksRefine(:,1:3);
s = 45;
m = 8;


sub = (r(:,2) > s + m) | (r(:,2) < s - m);
r(sub, :) = [];
figure(1)
hold on;
imagesc(squeeze(v(:,s,:)))
scatter(r(:,3),r(:,1),'MarkerEdgeColor',[0 .1 .5],...
              'LineWidth',1.5)
%%
voxelSize = 0.174;  % In Angstroms, just for plotting
merge_dist = 1.3
distMerge = merge_dist/voxelSize  % in voxels
minIntensityPeak = 70;
minPeakSigma     = 3;

% Main loop
% for a0 = 1:numLoops
% Merge peaks
% mergeAndThresholdAtomTraceHelper(sPeaks2,distMerge,voxelSize);
%%

sPeaks3 = mergeAndThresholdAtomTraceHelper(sPeaks2,distMerge,voxelSize);

r = sPeaks3.peaksRefine(:,1:3);
s = 45;
m = merge_dist/0.174


sub = (r(:,2) > s + m) | (r(:,2) < s - m);
r(sub, :) = [];
figure(1)
hold on;
imagesc(squeeze(v(:,s,:)))
scatter(r(:,3),r(:,1),'MarkerEdgeColor',[0 .1 .5],...
              'LineWidth',1.5)
%%
% Refine
sPeaks4 = refineSitesSubtractNeighborAtomTraceHelper(v,sPeaks3);
%%
sPeaks4 = refineSitesAtomTraceHelper(v,sPeaks4);
%%
voxelSize = 0.174;  % In Angstroms, just for plotting
merge_dist = 1.2
distMerge = merge_dist/voxelSize  % in voxels
minIntensityPeak = 25;
minPeakSigma     = 3;
sPeaks4 = mergeAndThresholdAtomTraceHelper(sPeaks4,distMerge,voxelSize);
del = ((sPeaks4.peaksRefine(:,4) ...
+ sPeaks4.peaksRefine(:,6) ...
< minIntensityPeak)) ...
| ...
(sPeaks4.peaksRefine(:,5) < minPeakSigma);
sPeaks4.peaksRefine(del,:) = [];
%%
r = sPeaks4.peaksRefine(:,1:3);
s = 50;
m = merge_dist/0.174
m = 8

sub = (r(:,2) > s + m) | (r(:,2) < s - m);
% r(sub, :) = [];
figure(1)
hold on;
imagesc(squeeze(v(:,s,:)))
scatter(r(:,3),r(:,1),'MarkerEdgeColor',[0 .1 .5],...
              'LineWidth',1.5)
%%
figure(2)
histogram(sPeaks4.peaksRefine(:,4),100)
%%
figure(3)
histogram(sPeaks4.peaksRefine(:,5),100)
%%          
minIntensityPeak = 40
minPeakSigma = 3.05
% Delete peaks using extra criteria
del = ((sPeaks4.peaksRefine(:,4) ...
+ sPeaks4.peaksRefine(:,6) ...
< minIntensityPeak)) ...
| ...
(sPeaks4.peaksRefine(:,5) < minPeakSigma);
sPeaks4.peaksRefine(del,:) = [];

r = sPeaks4.peaksRefine(:,1:3);
s = 45;
m = merge_dist/0.174


sub = (r(:,2) > s + m) | (r(:,2) < s - m);
r(sub, :) = [];
figure(1)
hold on;
imagesc(squeeze(v(:,s,:)))
scatter(r(:,3),r(:,1),'MarkerEdgeColor',[0 .1 .5],...
              'LineWidth',1.5)
%%
% Remove peaks with regular criteria
radiusNN         = 3/0.174;   % in pixels / voxels
minNumNNsAllowed = 5;  % if site has less NNs than this, it will be removed
maxNNbins        = 25;  % histogram
flagLocalIntensityCriteria = false;
% low local intensity ratio
distCheck        = 8;
numNNcheck       = 16;
minIntRatio      = 0.25;
removeSitesAtomTraceHelper(sPeaks4, flagLocalIntensityCriteria, radiusNN, ...
    minNumNNsAllowed, maxNNbins, distCheck, numNNcheck, minIntRatio);
%%

sPeaks5 = removeSitesAtomTraceHelper(sPeaks4, flagLocalIntensityCriteria,...
    radiusNN, minNumNNsAllowed, maxNNbins, ...
    distCheck, numNNcheck, minIntRatio);
% if a0 > (numLoops * 0.5)
% Check local intensity ratio
% sPeaks2 = removeSitesAtomTraceHelper(sPeaks2,true);
% end
% Refine
sPeaks5 = refineSitesSubtractNeighborAtomTraceHelper(v,sPeaks5);

Np = size(sPeaks5.peaksRefine,1);
disp(['        iter # ' num2str(0) '/' num2str(0) ... 
' done, total number of sites = ' num2str(Np)])
% end

r = sPeaks5.peaksRefine(:,1:3);
s = 40;
m = merge_dist/0.174


sub = (r(:,2) > s + m) | (r(:,2) < s - m);
r(sub, :) = [];
figure(1)
hold on;
imagesc(squeeze(v(:,s,:)))
scatter(r(:,3),r(:,1),'MarkerEdgeColor',[0 .1 .5],...
              'LineWidth',1.5)
%%
r = sPeaks5.peaksRefine(:,1:3);
s = 55;
m = 10;


sub = (r(:,2) > s + m) | (r(:,2) < s - m);
r(sub, :) = [];
figure(1)
hold on;
imagesc(squeeze(v(:,s,:)))
scatter(r(:,3),r(:,1),'MarkerEdgeColor',[0 .1 .5],...
              'LineWidth',1.5)
%%
figure(2)
scatter3(r(:,3),r(:,1),r(:,2))









