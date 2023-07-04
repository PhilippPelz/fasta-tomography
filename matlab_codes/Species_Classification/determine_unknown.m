function Flag = determine_unknown(Model,Atomtype,DistTh,NcoordTh,RatioTh)

    Flag = zeros(1,size(Model,2));
    Coord = zeros(1,size(Model,2));
    RatioAr = zeros(1,size(Model,2));


    for i=1:size(Model,2)
      
        
        
        curr_pos = Model(:,i);
        Dist = sqrt(sum((Model - repmat(curr_pos,[1 size(Model,2)])).^2,1));
        Dist(i) = 1000000;
        %DistThind = find(Dist < DistTh);
        
        N_atomtype = Atomtype(Dist < DistTh);
        Ratio = log(sum(N_atomtype==2) / sum(N_atomtype==1))/log(2);

        %N_model = Model(:,DistThind);
        
        if Atomtype(i) == 1
            if length(N_atomtype) < NcoordTh 
                Flag(i) = 1;
            elseif Ratio < log(RatioTh)/log(2)
                Flag(i) = 1;
            end
        end
        Coord(i) = length(N_atomtype);
        RatioAr(i) = Ratio;
    end
    
end
            

    