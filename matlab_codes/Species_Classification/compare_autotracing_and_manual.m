Man = importdata('M_20151006_NAcube_2ndtry_NAlist_temp.mat');
Tra = importdata('Sdenoise_all_2_Dset_old_151004_NAcube_NAlist_th_noex_step1.mat');
%%
SameInd =[];
DiffInd=[];
for i=1:size(Man.model,2)
    curr_model = Man.model(:,i);
    Dist = sqrt(sum((Man.new_model-repmat(curr_model,[1 size(Man.new_model,2)])).^2,1));
    if min(Dist)<0.0001
        SameInd(end+1)=i;
    else
        DiffInd(end+1)=i;
    end
end