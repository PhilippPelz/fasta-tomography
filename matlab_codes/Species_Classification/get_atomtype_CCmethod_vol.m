function [atomtype] =get_atomtype_CCmethod_vol(DataMatrix, atom_pos, avatomFe, avatomPt, avatomNA, boxhalfsize)

atomtype = zeros(1,size(atom_pos,2));

for kkk=1:size(atom_pos,2)
    curr_x = round(atom_pos(1,kkk));
    curr_y = round(atom_pos(2,kkk));
    curr_z = round(atom_pos(3,kkk));
    
    curr_vol = DataMatrix(curr_x-boxhalfsize:curr_x+boxhalfsize,...
                          curr_y-boxhalfsize:curr_y+boxhalfsize,...
                          curr_z-boxhalfsize:curr_z+boxhalfsize);
    
    D_Fe= sum(abs(curr_vol(:)-avatomFe(:)));   % Rfactor N=1  ; all elements > 0 
    D_Pt = sum(abs(curr_vol(:)-avatomPt(:)));
    D_NA = sum(abs(curr_vol(:)-avatomNA(:)));
    
    D_ar = [D_Fe; D_Pt; D_NA];
    
    [Min, MinInd] = min(D_ar);
    
    if MinInd == 1
        atomtype(kkk) = 1;
    elseif MinInd == 2
        atomtype(kkk) = 2;
    elseif MinInd == 3
        atomtype(kkk) = -1;
    end
end
   
fprintf('number of Fe: \t %i atoms\n', sum(atomtype==1))
fprintf('number of Pt: \t %i atoms\n', sum(atomtype==2))
fprintf('number of NA: \t %i atoms\n', sum(atomtype==-1))
