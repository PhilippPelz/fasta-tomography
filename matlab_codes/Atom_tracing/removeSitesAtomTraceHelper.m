function [r1] = removeSitesAtomTraceHelper(r, volsize,flagLocalIntensityCriteria, radiusNN, ...
    minNumNNsAllowed, maxNNbins, distCheck, numNNcheck, minIntRatio)
% Colin Ophus - April 2018 - atom tracing for phase contrast reconstructions
% 35 - use nearest neighbor criteria to remove atomic sites - for example
% vacuum atoms. If no output is specified, compute NN histogram.
% If secondary flag is enabled, use local intensity criteria.

% Low density crteria


% Generate NN histogram
Np = size(r,1);
r2max = radiusNN^2;
if nargout == 0
    bins = (0:maxNNbins)';
    Nb = length(bins);
    counts = zeros(Nb,1);
    skip = 1;
    for a0 = 1:skip:Np
        d2 =  (r(a0,1) - r(:,1)).^2 ...
            + (r(a0,2) - r(:,2)).^2 ...
            + (r(a0,3) - r(:,3)).^2;
        numNN = sum(d2<r2max) - 1;
        counts(min(numNN + 1,Nb)) = counts(min(numNN + 1,Nb)) + 1;
        
    end
    
    figure(33)
    clf
    scatter(bins,counts,'marker','d','sizedata',40,...
        'markeredgecolor',[1 0 0],'markerfacecolor',[1 0 0])
    xlabel(['Number of NNs < ' sprintf('%.02f',radiusNN) ' voxels'])
    drawnow;
end


peaksRefine = r;
if (nargin == 1) || (flagLocalIntensityCriteria == false)
    % If needed, delete atoms due to low density
    del = false(Np,1);
    if nargout > 0
        for a0 = 1:Np
            d2 =  (peaksRefine(a0,1) - peaksRefine(:,1)).^2 ...
                + (peaksRefine(a0,2) - peaksRefine(:,2)).^2 ...
                + (peaksRefine(a0,3) - peaksRefine(:,3)).^2;
            numNN = sum(d2<r2max) - 1;
            if numNN < minNumNNsAllowed
                del(a0) = true;
            end
        end
    end
    
else
    
    
    % set up NN network
    volInds = zeros(volsize);
    inds = sub2ind(volsize,...
        round(r(:,1)),...
        round(r(:,2)),...
        round(r(:,3)));
    volInds(inds) = 1:Np;
    
    % init
    NNintensityRatio = zeros(Np,1);
    
    % loop over sites
    vCut = (floor(-distCheck-1):ceil(distCheck+1))-1;
    for a0 = 1:Np
        volCrop = volInds( ...
            mod(round(r(a0,1))+vCut,sPeaks.volSize(1))+1,...
            mod(round(r(a0,2))+vCut,sPeaks.volSize(2))+1,...
            mod(round(r(a0,3))+vCut,sPeaks.volSize(3))+1);
        indsCheck = volCrop(volCrop>0);
        
        d2 =  (r(a0,1) - r(indsCheck,1)).^2 ...
            + (r(a0,2) - r(indsCheck,2)).^2 ...
            + (r(a0,3) - r(indsCheck,3)).^2;
        [~,inds] = sort(d2);
        
        if length(inds) > 1
            indsNN = inds(2:min(numNNcheck+1,length(inds)));
            indsAll = indsCheck(indsNN);
            NNintensityRatio(a0) = ...
                r(a0,4) .* r(a0,5) ...
                / mean(r(indsAll,4) ...
                .* r(indsAll,5));
        end
    end
    
    
    del = NNintensityRatio < minIntRatio;
    
end

peaksRefine(del,:) = [];
r1 = peaksRefine;
disp([num2str(sum(del)) ' atoms removed from ' num2str(Np) ' total'])


end