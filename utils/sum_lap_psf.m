function [Sum_psf,sum_diff_b] = sum_lap_psf(size, b)
    center = (size+1)/2;
    X = -size + center:size - center;
    Y = -size + center:size - center;
    
    %% PSF sum
    kernel = ones(size, size);
    for ii=1:size
        for jj=1:size
           kernel(ii,jj) = (b^2 / 4) * exp(-b*(abs(X(ii)) + abs(Y(jj))));
        end
    end
    % let compute the sum
    Sum_psf = sum(kernel(:));
    
    %% differential with respect to b
    difflap_b = zeros(size, size);
    for ii=1:size
        for jj=1:size
            var1 = (2*b - b^2 *(abs(X(ii)) + abs(Y(jj)))) / 4;
            var2 = exp(-b*(abs(X(ii)) + abs(Y(jj))));
            difflap_b(ii,jj) = var1 * var2;
        end
    end
    % let compute the sum
    sum_diff_b = sum(difflap_b(:));
    
end