function [atom_pos, close_pos, stats, statsI, CurrData] = find_possible_atoms(DataMatrix, Dmin, MaxNumberAtoms, BoxSize0,BoxSize1,BoxSize2,Th)

[box1coordinates, BoxCenter1, BoxRadius1, sphere1] = create_box(BoxSize1);
[box2coordinates, BoxCenter2, BoxRadius2, sphere2] = create_box(BoxSize2);
% BoxRadius2 = ceil(BoxRadius2);
atom_pos=zeros(3,MaxNumberAtoms);   %store position of peaks
close_pos=[];                       %store position of peaks that are too close
nclose=0;                           %counter for too close atoms
i_atom=0;                           %index for current atom
[Xsize, Ysize, Zsize] = size(DataMatrix);


CurrData=DataMatrix;                %Matrix used for Gaussian substraction
M=find_maxima3D(CurrData,sphere1);  %find all maxima

noatomM = zeros(size(DataMatrix));

%continue this loop basicaly forever
%"inf" in a for loop is ugly, therefore 1e8
for iteration = 1:1e8
    
    i_atom=i_atom+1;
    
    %show status
    fprintf('Atom number: %04i\n',i_atom)
    if mod(i_atom,100)==0
        ¾¾
    end
    
    % find coordinates of peaks
    ind_find=find(M);
     if isempty(ind_find)
        break
     end   
      [xx,yy,zz] = ind2sub(size(M),ind_find);        
        
    % find peaks and sort them
    [peak_sort, ind_sort] = get_peak_values(CurrData,xx,yy,zz,ind_find);
    
    %take highest peak
    hind=ind_sort(1); hpeak=peak_sort(1);
    
    %peak location
    x=xx(hind);y=yy(hind);z=zz(hind);
    
    %Get Data ROI
    [x y z];
    DataBox1 = get_data_box(CurrData,x,y,z,BoxRadius1);
    
    %start values and bounds for fitting
    ymax_init       = DataBox1(BoxCenter1,BoxCenter1,BoxCenter1);
    fit_param_init  = [0,  ymax_init, 0,  0,  0,  0.2,  0.2,  0.8,  0.0,  0.0,  0.0];
    fixed           = [0,  0        ,  0,  0,  0,    0,    0,    0,    0,    0,    0];
    lb              = [0,  0        , -1*BoxRadius1, -1*BoxRadius1, -1*BoxRadius1,    0,    0,    0, -pi, 0, -pi];
    ub              = [inf,inf      ,  BoxRadius1,  BoxRadius1,  BoxRadius1,  inf,  inf,  inf, pi, pi,  pi];
    
    %perform the fit
    [fit_resultI,resnorm,residual] = fit_gauss3D_PD(fit_param_init, box1coordinates, DataBox1, fixed, lb, ub);
 
    %calc position of new atom and FWHM
    new_x = x+fit_resultI(3);
    new_y = y+fit_resultI(4);
    new_z = z+fit_resultI(5);
    
    %integer new position
    rx=round(new_x);
    ry=round(new_y);
    rz=round(new_z);
    
    %sub pixel offset to integer position
    dx=new_x-rx;
    dy=new_y-ry;
    dz=new_z-rz;
    
    %get minimum distance
    if i_atom>1
        d0 = get_minimum_distance(new_x,new_y,new_z,atom_pos(:,1:i_atom-1));
    else
        d0 = inf;
    end
    
    valid_atom=1;
    
    %local area used to calculate maxima
    roisize=25; 
    roihalf=(roisize-1)/2;
    xhi=new_x+roihalf;
    yhi=new_y+roihalf;
    zhi=new_z+roihalf;

    xlo=new_x-roihalf;
    ylo=new_y-roihalf;
    zlo=new_z-roihalf;
    
    if xhi > Xsize || yhi > Ysize || zhi > Zsize || xlo < 1 || ylo < 1 || zlo < 1
        fprintf('Atom number: %04i \t outside volume!\n',i_atom)
        valid_atom=0;
    end
    
    %check minimum distance constraint
    if d0<Dmin
        fprintf('Atom number: %04i \t Too close!\n',i_atom)
        valid_atom=0;
    end
    
    %check, if new fit position is out of bounce
    %should not happen, if fit bounds are set correctly
    if new_x>Xsize || new_y>Ysize || new_z>Zsize || new_x<1 || new_y<1 || new_z<1
        fprintf('Atom number: %04i: \t Out of Border!\n',i_atom)
        valid_atom=0;
    end
    
    
    if valid_atom==0     %atom is to close
        
        
        nclose=nclose+1;
        close_pos(:,nclose)=[new_x new_y new_z];
        
        M(x,y,z)=0;       %remove this peak from the list of potential atoms
        i_atom=i_atom-1;  %decrease atom counter, since it is not an atom
        noatomM(x,y,z) = 1;
    
    else                  %yes, it is a valid peak
        
        %substract fit from data
        
        fit_resultI(3:5)=[dx dy dz];                           %use subpixel offsets
        FitBox2 = calc_gauss3D_PD(fit_resultI, box2coordinates);  %calculate gaussian
        FitBox2 = FitBox2-fit_resultI(1);                      %substract background
        
        DataBox2 = get_data_box(CurrData,rx,ry,rz,BoxRadius2); %get data
        DiffBox2 = DataBox2-FitBox2;                           %substact gaussian fit
        DiffBox2 = max(0,DiffBox2);                            %remove negative values
        
        %update data for next iteration
        CurrData(rx-BoxRadius2:rx+BoxRadius2,ry-BoxRadius2:ry+BoxRadius2,rz-BoxRadius2:rz+BoxRadius2)=DiffBox2;
                   
        if mod(i_atom,10) == 0    
            m = 6;
            s = round(size(CurrData,2)/2)-8*m;            
        
            figure(1)
            clf
            
            subplot(2,2,1)
            hold on;
            r = atom_pos(1:3,1:i_atom);
            sub = (r(2,:) > s + m) | (r(2,:) < s - m);
            r(:, sub) = [];
            im = squeeze(sum(CurrData(:,s-m:s+m,:),2));
            imagesc(im)
            scatter(r(3,:),r(1,:),'MarkerEdgeColor',[1 0 0],...
                          'LineWidth',1.5)
                      
            s = s + 4*m;
            subplot(2,2,2)
            hold on;
            r = atom_pos(1:3,1:i_atom);
            sub = (r(2,:) > s + m) | (r(2,:) < s - m);
            r(:, sub) = [];
            im = squeeze(sum(CurrData(:,s-m:s+m,:),2));
            imagesc(im)
            scatter(r(3,:),r(1,:),'MarkerEdgeColor',[1 0 0],...
                          'LineWidth',1.5)
                      
            s = s + 4*m;
            subplot(2,2,3)
            hold on;            
            r = atom_pos(1:3,1:i_atom);
            sub = (r(2,:) > s + m) | (r(2,:) < s - m);
            r(:, sub) = [];
            im = squeeze(sum(CurrData(:,s-m:s+m,:),2));
            imagesc(im)
            scatter(r(3,:),r(1,:),'MarkerEdgeColor',[1 0 0],...
                          'LineWidth',1.5)
                      
            s = s + 4*m;
            subplot(2,2,4)
            hold on;            
            r = atom_pos(1:3,1:i_atom);
            sub = (r(2,:) > s + m) | (r(2,:) < s - m);
            r(:, sub) = [];
            im = squeeze(sum(CurrData(:,s-m:s+m,:),2));
            imagesc(im)
            scatter(r(3,:),r(1,:),'MarkerEdgeColor',[1 0 0],...
                          'LineWidth',1.5)
                     

    %         axis equal off
    %         colormap(hot(256))
            caxis([min(im(:)) max(im(:))])
            drawnow         
        end
        
        %save statistics
        statsI.resnorm(i_atom) = resnorm;
        statsI.residual{i_atom} = residual(:);
        statsI.fit{i_atom}=fit_resultI;
        statsI.peak_bg(i_atom)=fit_resultI(1);
        statsI.peak_hight(i_atom)=fit_resultI(2);
        statsI.peak_FWHM(:,i_atom)=2.3548./(sqrt(2)*fit_resultI(6:8));
        statsI.localmax_pos{i_atom} = [x y z];

        
        %save atom position
        atom_pos(1,i_atom) = new_x;
        atom_pos(2,i_atom) = new_y;
        atom_pos(3,i_atom) = new_z;
        
%         atom_pos(:,i_atom)
        
        fit_result  = stats_cal_PD(DataMatrix,atom_pos(:,i_atom),atom_pos(:,i_atom),BoxSize1);
          
        stats.fit{i_atom}=fit_result;
        stats.peak_bg(i_atom)=fit_result(1);
        stats.peak_hight(i_atom)=fit_result(2);
        stats.peak_FWHM(:,i_atom)=2.3548./(sqrt(2)*fit_result(6:8)); 
        stats.localmax_pos{i_atom} = [x y z];
          
        %find maxima again, but restrict to the region next to the
        %currently substracted atom
        
        %local area used to calculate maxima
        roisize=25; 
        roihalf=(roisize-1)/2;
        xhi=rx+roihalf;
        yhi=ry+roihalf;
        zhi=rz+roihalf;
        
        xlo=rx-roihalf;
        ylo=ry-roihalf;
        zlo=rz-roihalf;
        
        
        %get local density inormation
        CurrDataRoi=CurrData(xlo:xhi,ylo:yhi,zlo:zhi);
        
        %find maxima
        MRoi=find_maxima3D(CurrDataRoi,sphere1);
        
        %remove the a 2 pixel border (there might be artifical maxima)
        roiborder=2;
        MRoi=MRoi(1+roiborder:roisize-roiborder,1+roiborder:roisize-roiborder,1+roiborder:roisize-roiborder);
        
        %update maxima matrix around the current atom
        M(xlo+roiborder:xhi-roiborder,ylo+roiborder:yhi-roiborder,zlo+roiborder:zhi-roiborder)=MRoi;
        
        %do not trace current atom again
        M(rx-2:rx+2,ry-2:ry+2,rz-2:rz+2)=0; 
        noatomM(rx-2:rx+2,ry-2:ry+2,rz-2:rz+2)=1; 
                
        M(noatomM==1) = 0;
        
        Peaks = DataMatrix;
        Peaks(M<1) = 0;
        M(Peaks < Th) = 0;
        
    end
    
    if i_atom==MaxNumberAtoms
        break
    end
    
end
