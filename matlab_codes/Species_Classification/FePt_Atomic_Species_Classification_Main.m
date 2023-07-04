%% load data

% 3D reconstruction volume
RecVol = importdata('FePt_reconstruction_volume.mat');

% loose support used for peak tracing (see FePt_Peak_Tracing_Main.m)
LooseSupport = importdata('Loose_Support.mat');

% traced peak positions (see FePt_Peak_Tracing_Main.m) after manual
% adjustment (the output varialble atom_pos from FePt_Peak_Tracing_Main.m)
atom_pos = importdata('Traced_peak_pos.mat');

% the final Gaussian peak fitting result (see FePt_Peak_Tracing_Main.m) after
% manual adjustment (the output varialble stats from FePt_Peak_Tracing_Main.m)
new_stats = importdata('Traced_peak_stats.mat');

% pad ESTvolume to match the peak tracing (similar padding is done for peak tracing)
RecVol = RecVol .* LooseSupport;

% padd zero
RecVol_padded = zeros(276,276,276);
RecVol_padded(11:266,11:266,11:266) = RecVol;

% flag for plotting during the procedure
PLOT_YN = 0;

nonatom_lower = 0;
nonatom_upper = 0.1;

TH = 20;

TH_dubious = 15;
%% obtain global intensity histogram

curr_model = atom_pos;

% initialize array for integrated intensity
intensity_5x5 = zeros(1,size(curr_model,2));
intensity_3x3 = zeros(1,size(curr_model,2));

% integrate intensity (for 3x3x3 and 5x5x5 voxels) for each traced peak
for j=1:size(curr_model,2)
    curr_pos = round(curr_model(:,j));
    box_3x3 = RecVol_padded(curr_pos(1)-1:curr_pos(1)+1,curr_pos(2)-1:curr_pos(2)+1,curr_pos(3)-1:curr_pos(3)+1);
    box_5x5 = RecVol_padded(curr_pos(1)-2:curr_pos(1)+2,curr_pos(2)-2:curr_pos(2)+2,curr_pos(3)-2:curr_pos(3)+2);
    intensity_3x3(j) = sum(box_3x3(:));
    intensity_5x5(j) = sum(box_5x5(:));
end

% histogram of integrated intensity
[hist_3x3, cen_3x3] = hist(intensity_3x3,100);
[hist_5x5, cen_5x5] = hist(intensity_5x5,100);
    
initcen_3x3 = [15 25]; initpeak_3x3 = [400 650]; initwidth_3x3 = [3 3];
initcen_5x5 = [45 70]; initpeak_5x5 = [5 max(hist_5x5)]; initwidth_5x5 = [5 10];

% plot histogram if PLOT_YN flag is 1
if PLOT_YN
    figure;
    i_guess = [0 initpeak_3x3(1) initcen_3x3(1) initwidth_3x3(1) 0 initpeak_3x3(2) initcen_3x3(2) initwidth_3x3(2)];
    Xdata = cen_3x3;
    Ydata = hist_3x3; Ydata(1:20) = 0;
    [p, fminres, fitresult] = My_two_gaussianfit(Xdata, Ydata, i_guess);  



    hist(intensity_3x3,100);
    hold on
    plot(cen_3x3, fitresult, 'r-', 'LineWidth',3);
    hold off   
    xlabel('integrated intensity (a.u.)');
    ylabel('# atoms');
    title(sprintf('3x3x3 cube, pos [%5.2f, %5.2f]\n width [%5.2f, %5.2f] height [%5.2f, %5.2f]',...
          p(3), p(7), abs(p(4)), abs(p(8)), p(2), p(6)));
end
    



%% compute initial average Fe atom, Pt atom and non atom

% find indices of potential Fe, Pt, non atom from the given threshold
defFeind = find(intensity_3x3<TH & intensity_3x3 > nonatom_upper);
defPtind = find(intensity_3x3>TH );
defNAind = find(intensity_3x3<nonatom_upper & intensity_3x3>nonatom_lower);

% compute average atom from the given indices
[avatomFe]= compute_average_atom_from_vol(RecVol_padded,atom_pos,defFeind,2);
[avatomPt]= compute_average_atom_from_vol(RecVol_padded,atom_pos,defPtind,2);
[avatomNA]= compute_average_atom_from_vol(RecVol_padded,atom_pos,defNAind,2);

% assign atomtype, 1 = Fe, 2 = Pt, -1 = non atom
atomtype = zeros(1,length(intensity_3x3));
atomtype(defFeind) = 1;
atomtype(defPtind) = 2;
atomtype(defNAind) = -1;

%% atom classification iteration
curr_iternum = 0;

exitFlag = 0;

% iterate until there's no change in atomic specise classification
while ~exitFlag
    if PLOT_YN

        subplot(4,1,1)
        [~,cen_3x3_total ]= hist(intensity_3x3,100);


        hist(intensity_3x3,100);

        hold all
        plot(cen_3x3, fitresult, 'r-', 'LineWidth',3);

        plot(cen_3x3, (p(1)+p(5))/2+p(2)*exp(-((Xdata-p(3))/p(4)).^2),'g-','LineWidth',2);
        plot(cen_3x3, (p(1)+p(5))/2+p(6)*exp(-((Xdata-p(7))/p(8)).^2),'g-','LineWidth',2); 
        ylim([0 800]);
        xlabel('integrated intensity (a.u.)');
        ylabel('# atoms');
        title(sprintf('3x3x3 cube, pos [%5.2f, %5.2f]\n width [%5.2f, %5.2f] height [%5.2f, %5.2f]\n iteration %d',...
              p(3), p(7), abs(p(4)), abs(p(8)), p(2), p(6),curr_iternum));

    end

    intensity_3x3_Fe = intensity_3x3(atomtype==1);
    intensity_3x3_Pt = intensity_3x3(atomtype==2);
    intensity_3x3_NA = intensity_3x3(atomtype==-1);

    if PLOT_YN
        subplot(4,1,2)
        hist(intensity_3x3_Fe,cen_3x3_total ); ylim([0 600]);
        xlabel('integrated intensity (a.u.)');
        ylabel('# atoms');
        title(sprintf('%d Fe atoms',sum(atomtype==1)));
        ylim([0 800]);

        subplot(4,1,3)
        hist(intensity_3x3_Pt,cen_3x3_total )
        xlabel('integrated intensity (a.u.)');
        ylabel('# atoms');
        title(sprintf('%d Pt atoms',sum(atomtype==2)));
        ylim([0 800]);

        subplot(4,1,4)
        hist(intensity_3x3_NA,cen_3x3_total )
        xlabel('integrated intensity (a.u.)');
        ylabel('# atoms');
        title(sprintf('%d NA atoms',sum(atomtype==-1)));
        ylim([0 800]);

        set(gcf,'PaperPositionMode','auto')
    end

    old_atomtype = atomtype;
    % obtain updated atom classification by comparing each peaks with average atom species
    [atomtype] = get_atomtype_CCmethod_vol(RecVol_padded, atom_pos, avatomFe, avatomPt, avatomNA, 2);

    % re-compute average atomic species from the updated atom classification
    [avatomFe]= compute_average_atom_from_vol(RecVol_padded,atom_pos,find(atomtype==1),2);
    [avatomPt]= compute_average_atom_from_vol(RecVol_padded,atom_pos,find(atomtype==2),2);
    [avatomNA]= compute_average_atom_from_vol(RecVol_padded,atom_pos,defNAind,2);

    curr_iternum = curr_iternum + 1

    % if there is no change in the atomic specise classification, turn on
    % the exitflag
    if sum(old_atomtype ~= atomtype) == 0
        exitFlag = 1;
    end
end

%% insert atom based on R factor

% compute average Fe atom again from the previous classification result
[avatomFe3]= compute_average_atom_from_vol(RecVol_padded, atom_pos, find(atomtype==1), 2);

% compare each peak with average Fe atom with non-atom and mark peaks closer to non-atom
% to zero
[atomtype1, Rs] =insert_atoms_based_on_rfactor_PD_YY_bgsub(RecVol_padded, atom_pos, new_stats, avatomFe3,2);

%% apply manual adjustment of "atomtype1" to make physically reasonable model
atomtype = importdata('manually_adjusted_atomtype.mat');


%% remove non-atoms, re-run the classification of Pt and Fe

% take only Fe and Pt atoms, not non-atoms
curr_model = atom_pos(:,atomtype>0);
Cmodel = curr_model;
F_atomtype = zeros(1,size(curr_model,2));

% initialize variables
intensity_5x5 = zeros(1,size(curr_model,2));
intensity_3x3 = zeros(1,size(curr_model,2));

% re-calculate integrated intensities
for j=1:size(curr_model,2)
    curr_pos = round(curr_model(:,j));
    box_3x3 = RecVol_padded(curr_pos(1)-1:curr_pos(1)+1,curr_pos(2)-1:curr_pos(2)+1,curr_pos(3)-1:curr_pos(3)+1);
    box_5x5 = RecVol_padded(curr_pos(1)-2:curr_pos(1)+2,curr_pos(2)-2:curr_pos(2)+2,curr_pos(3)-2:curr_pos(3)+2);
    intensity_3x3(j) = sum(box_3x3(:));
    intensity_5x5(j) = sum(box_5x5(:));
end

% find indices of potential Fe, Pt from the given threshold
defFeind = find(intensity_3x3<TH);
defPtind = find(intensity_3x3>TH);

%% compute average Fe atom and remove non-atoms
% compute average atom
[avatomFe]= compute_average_atom_from_vol(RecVol_padded,curr_model,defFeind,2);
[avatomPt]= compute_average_atom_from_vol(RecVol_padded,curr_model,defPtind,2);

atomtype = zeros(1,length(intensity_3x3));
atomtype(defFeind) = 1;
atomtype(defPtind) = 2;


%% atom classification iteration of Fe and Pt
curr_iternum = 0;
exitFlag = 0;

while ~exitFlag
    
    if PLOT_YN == 1
        subplot(3,1,1)
        [~,cen_3x3_total ]= hist(intensity_3x3,100);

        hist(intensity_3x3,100);

        hold all
        plot(cen_3x3, fitresult, 'r-', 'LineWidth',3);

        plot(cen_3x3, (p(1)+p(5))/2+p(2)*exp(-((Xdata-p(3))/p(4)).^2),'g-','LineWidth',2);
        plot(cen_3x3, (p(1)+p(5))/2+p(6)*exp(-((Xdata-p(7))/p(8)).^2),'g-','LineWidth',2); 
        ylim([0 700]);
        xlabel('integrated intensity (a.u.)');
        ylabel('# atoms');
        title(sprintf('3x3x3 cube, pos [%5.2f, %5.2f]\n width [%5.2f, %5.2f] height [%5.2f, %5.2f]\n iteration %d',...
              p(3), p(7), abs(p(4)), abs(p(8)), p(2), p(6),curr_iternum));


        intensity_3x3_Fe = intensity_3x3(atomtype==1);
        intensity_3x3_Pt = intensity_3x3(atomtype==2);

        subplot(3,1,2)
        hist(intensity_3x3_Fe,cen_3x3_total ); ylim([0 90]);
        xlabel('integrated intensity (a.u.)');
        ylabel('# atoms');
        title(sprintf('%d Fe atoms',sum(atomtype==1)));
        ylim([0 700]);

        subplot(3,1,3)
        hist(intensity_3x3_Pt,cen_3x3_total )
        xlabel('integrated intensity (a.u.)');
        ylabel('# atoms');
        title(sprintf('%d Pt atoms',sum(atomtype==2)));
        ylim([0 700]);
    end
    
    old_atomtype = atomtype;

    % obtain updated atom classification by comparing each peaks with average atom species
    [atomtype] = get_atomtype_CCmethod_vol_twoatom(RecVol_padded, curr_model, avatomFe, avatomPt, 2);

    % re-compute average atomic species from the updated atom classification
    [avatomFe]= compute_average_atom_from_vol(RecVol_padded,curr_model,find(atomtype==1),2);
    [avatomPt]= compute_average_atom_from_vol(RecVol_padded,curr_model,find(atomtype==2),2);

    curr_iternum = curr_iternum + 1
    
    % if there is no change in the atomic specise classification, turn on
    % the exitflag
    if sum(old_atomtype ~= atomtype) == 0
        exitFlag = 1;
    end
end

%% 

% get new model of Fe and Pt
new_model = Cmodel;
new_atomtype = atomtype;

% Determine missing wedge atoms based on local Fe coordination number
% Looking at overal FePt volume, there are approximately 2 times more Pt
% atoms than Fe atoms. Therefore, any atoms which shows more Fe than Pt
% atoms in local coordinates are physically not likely, due to missing
% wedge problem.
Flag = determine_unknown(new_model,new_atomtype,8.8601,8.1,2);
%sum(Flag==1)

%% prepare next step of determining dubious atoms near missing wedge direction

% model cooordinates with good atom (non-missing wedge)
Kmodel = new_model(:,Flag==0);

% model cooordinates with undetermined atom (missing wedge)
Umodel = new_model(:,Flag==1);



%% prepare variables for dubious missing wedge atom classification
curr_model = Umodel;
atom_pos = Umodel;
model = Umodel;

% re-calculate integrated intensity
intensity_5x5 = zeros(1,size(curr_model,2));
intensity_3x3 = zeros(1,size(curr_model,2));

for j=1:size(curr_model,2)
    curr_pos = round(curr_model(:,j));
    box_3x3 = RecVol_padded(curr_pos(1)-1:curr_pos(1)+1,curr_pos(2)-1:curr_pos(2)+1,curr_pos(3)-1:curr_pos(3)+1);
    box_5x5 = RecVol_padded(curr_pos(1)-2:curr_pos(1)+2,curr_pos(2)-2:curr_pos(2)+2,curr_pos(3)-2:curr_pos(3)+2);
    intensity_3x3(j) = sum(box_3x3(:));
    intensity_5x5(j) = sum(box_5x5(:));
end

% find inital atom indices for Fe and Pt
defFeind = find(intensity_3x3<TH_dubious);
defPtind = find(intensity_3x3>TH_dubious);


%% compute average Fe atom and remove non-atoms

% compute average atoms
[avatomFe]= compute_average_atom_from_vol(RecVol_padded,curr_model,defFeind,2);
[avatomPt]= compute_average_atom_from_vol(RecVol_padded,curr_model,defPtind,2);

% initialize atomtype
atomtype = zeros(1,length(intensity_3x3));
atomtype(defFeind) = 1;
atomtype(defPtind) = 2;

%% atom classification iteration for missing wedge atoms

curr_iternum = 0;
exitFlag = 0;
while ~exitFlag

    if PLOT_YN ==1
        subplot(3,1,1)
        [~,cen_3x3_total ]= hist(intensity_3x3,100);


        hist(intensity_3x3,100);

        hold all
        plot(cen_3x3, fitresult, 'r-', 'LineWidth',3);

        plot(cen_3x3, (p(1)+p(5))/2+p(2)*exp(-((Xdata-p(3))/p(4)).^2),'g-','LineWidth',2);
        plot(cen_3x3, (p(1)+p(5))/2+p(6)*exp(-((Xdata-p(7))/p(8)).^2),'g-','LineWidth',2); 
        ylim([0 200]);
        xlabel('integrated intensity (a.u.)');
        ylabel('# atoms');
        title(sprintf('3x3x3 cube, pos [%5.2f, %5.2f]\n width [%5.2f, %5.2f] height [%5.2f, %5.2f]\n iteration %d',...
              p(3), p(7), abs(p(4)), abs(p(8)), p(2), p(6),curr_iternum));


        intensity_3x3_Fe = intensity_3x3(atomtype==1);
        intensity_3x3_Pt = intensity_3x3(atomtype==2);


        subplot(3,1,2)
        hist(intensity_3x3_Fe,cen_3x3_total ); ylim([0 200]);
        xlabel('integrated intensity (a.u.)');
        ylabel('# atoms');
        title(sprintf('%d Fe atoms',sum(atomtype==1)));
        ylim([0 200]);

        subplot(3,1,3)
        hist(intensity_3x3_Pt,cen_3x3_total )
        xlabel('integrated intensity (a.u.)');
        ylabel('# atoms');
        title(sprintf('%d Pt atoms',sum(atomtype==2)));
        ylim([0 200]);

        set(gcf,'PaperPositionMode','auto')
    end

    old_atomtype = atomtype;
    
     % obtain updated atom classification by comparing each peaks with average atom species   
    [atomtype] = get_atomtype_CCmethod_vol_twoatom_somefix(RecVol_padded, curr_model, avatomFe, avatomPt, 2, atomtype, []);

     % re-compute average atomic species from the updated atom classification
    [avatomFe]= compute_average_atom_from_vol(RecVol_padded,curr_model,find(atomtype==1),2);
    [avatomPt]= compute_average_atom_from_vol(RecVol_padded,curr_model,find(atomtype==2),2);

    curr_iternum = curr_iternum + 1
    
    % if there is no change in the atomic specise classification, turn on
    % the exitflag
    if sum(old_atomtype ~= atomtype) == 0
        exitFlag = 1;
    end
end

%% save results
new_atomtype(Flag==1) = atomtype;

save classified_atom_coordinates.mat new_model
save classified_atom_types.mat new_atomtype