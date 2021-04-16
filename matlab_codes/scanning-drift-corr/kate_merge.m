path ='/home/philipp/projects2/tomo/2019-09-09_kate_pd/';


fn1 = '0190909_BoxCd4_Pd_tomo3_rot0_norepeats.h5';

fn2 = '0190909_BoxCd4_Pd_tomo3_rot1_norepeats.h5';


d1 = h5read([path fn1],'/data');
d2 = h5read([path fn2],'/data');

s = size(d1);
ms = 1280;
merged = zeros(ms,ms,s(1));
refineMaxSteps = 30;
%%
scan_origins = zeros(s(2),2,2,s(1));
linear_drifts = zeros(2,1,s(1));

linear_drift = [0;-5.12];
% i = 1;
% sMerge = SPmerge01linear([0 90],squeeze(d1(i,:,:)),squeeze(d2(i,:,:)));
% sMerge = SPmerge02(sMerge, refineMaxSteps);
% imageFinal = SPmerge03(sMerge);
% merged(:,:,i) = imageFinal;

%%
for i = 1:s(1)
    i
    sMerge = SPmerge01linear([0 90],squeeze(d1(i,:,:)),squeeze(d2(i,:,:)));
%     sMerge = SPmerge02(sMerge, refineMaxSteps);
    imageFinal = SPmerge03(sMerge);
    merged(:,:,i) = imageFinal;
    scan_origins(:,:,:,i) = sMerge.scanOr;
    linear_drifts(:,:,i) = sMerge.xyLinearDrift;
    
end
%%
%%
for i = 1:s(1)
    i
    sMerge = SPmerge01linear([0 90],squeeze(d1(i,:,:)),squeeze(d2(i,:,:)));
%     sMerge = SPmerge02(sMerge, refineMaxSteps);
    imageFinal = SPmerge03(sMerge);
    merged(:,:,i) = imageFinal;
    scan_origins(:,:,:,i) = sMerge.scanOr;
    linear_drifts(:,:,i) = sMerge.xyLinearDrift;
    
end
%%
d.merged = merged;
d.scan_origins = scan_origins;

save([path 'merged_linear.mat'],'merged');



%%
% quickImages(merged,[-0.1 01])
%%
% imagesc(merged(:,:,16))