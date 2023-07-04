function [peak_sort, ind]= get_peak_values(CurrData,xx,yy,zz,ind)


        % get peak values
        for kkk=1:size(ind,1)
            peak(kkk) = sum(sum(sum(CurrData(xx(kkk)-1:xx(kkk)+1,yy(kkk)-1:yy(kkk)+1,zz(kkk)-1:zz(kkk)+1))));
        end
        
[peak_sort, ind] = sort(peak,'descend');
