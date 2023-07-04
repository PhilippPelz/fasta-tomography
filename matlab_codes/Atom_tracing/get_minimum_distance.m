function [minD, atom_number, pos_eigh] = get_minimum_distance(x,y,z,atom_pos)
%M. Bartels, UCLA, 2014

        dx = x - atom_pos(1,:);
        dy = y - atom_pos(2,:);
        dz = z - atom_pos(3,:);

        % minimal distance
        DD = sqrt(dx.^2 + dy.^2 + dz.^2);
        DD(DD==0) = 100;
        minD = min(DD);
        [no, atom_number] = find(DD==minD);
        pos_eigh = atom_pos(:,atom_number);
 
end


