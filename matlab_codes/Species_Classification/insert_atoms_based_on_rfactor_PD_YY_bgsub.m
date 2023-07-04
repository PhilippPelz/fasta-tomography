function [atomtype, Rs] =insert_atoms_based_on_rfactor_PD_YY_bgsub(Vol, atom_pos, stats, avatom, BoxHalfRad )

atomtype = -1*ones(1,size(atom_pos,2));
Rs = zeros(2,size(atom_pos,2));

box2coordinates  = create_box( BoxHalfRad*2+1);

fit_param_init  = [0,  max(avatom(:)),  0,  0,  0,  0.5,  0.5,    0.5,  0,  0,  0];
fixed           = [0,  0            ,  0,  0,  0,    0,    0,    0,    0,  0,    0];
lb              = [0,  0            , -2, -2, -2,    0,    0,    0, -pi,  0,  -pi];
ub              = [inf,inf          ,  2,  2,  2,  inf,  inf,  inf,  pi,  pi,  pi];

%perform the fit
[fit_result ] = fit_gauss3D_PD(fit_param_init, box2coordinates, avatom, fixed, lb, ub);
fit_result(1)
for kkk=1:size(atom_pos,2)
    
    
    
    curr_pos_r = round(atom_pos(:,kkk));
    DataBox = Vol(curr_pos_r(1)-BoxHalfRad:curr_pos_r(1)+BoxHalfRad,...
                 curr_pos_r(2)-BoxHalfRad:curr_pos_r(2)+BoxHalfRad,...
                 curr_pos_r(3)-BoxHalfRad:curr_pos_r(3)+BoxHalfRad);
             
    DataBox = DataBox - stats.fit{kkk}(1);
    ZeroBox = zeros(size(avatom));
    avatom_bgsub = avatom - fit_result(1);
    
    %calculate squared deviation ( = non-normalized r-factor)
    R1 = sum(abs(DataBox(:)-ZeroBox(:)));
    R2 = sum(abs(DataBox(:)-avatom_bgsub(:)));
    Rs(1,kkk) = R1;
    Rs(2,kkk) = R2;
    
    if (R1 > R2)
    
        %this is an atom, add to density matrix
        atomtype(kkk) = 1;
        
    else
        %not an atom, add background to density matrix
        atomtype(kkk) = 0;
    end
        
end

fprintf('number of inserted atoms: \t %i atoms\n', sum(atomtype==1))
fprintf('number of  skipped atoms: \t %i atoms\n', sum(atomtype==0))
%fprintf('percentage: \t\t\t\t %.1f percent atoms\n', 100*goodatomcount/(wrongatomcount+goodatomcount))