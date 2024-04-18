function [fftcomplet_laplace] = diff_laplace_b(im_shape,size, b)
difflap_b = zeros(size, size);
center = (size +1)/2;
x = -size+center:size-center;
y = -size+center:size-center;
[Sum_psf, sum_diff_b] = sum_lap_psf(size, b);
for ii=1:size
    for jj=1:size
        f = (b^2 / (4)) * exp(-b*(abs(x(ii)) + abs(y(jj))));
        var1 = (2*b - b^2 *(abs(x(ii)) + abs(y(jj)))) / 4;
        var2 = exp(-b*(abs(x(ii)) + abs(y(jj))));
        diff =  var1 * var2;
        difflap_b(ii,jj) = (diff * Sum_psf - f * sum_diff_b) / Sum_psf^2;
    end
end
    
%% Resize the gradient w.r.t. b
    fftcomplet_laplace = resize(difflap_b, im_shape);
end