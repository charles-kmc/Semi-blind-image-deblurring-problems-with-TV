function [lap] = psf_laplace(size, b)
    center  = (size+1) / 2;
    x = -size + center:size - center;
    y = -size + center:size - center;
    lap = zeros(size, size);
    for ii=1:size
        for jj=1:size
           lap(ii,jj) = (b^2 / 4) * exp(-b*(abs(x(ii)) + abs(y(jj))));
        end
    end
     lap = lap / sum(lap(:));
   
    end