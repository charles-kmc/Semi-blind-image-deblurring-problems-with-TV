function [sum_psf, sum_diffw1, sum_diffw2] = Sum_gauss_psf(taille, w1, w2, phi)
    center = (taille + 1) / 2;
    
    %% x and y vector
    x = -taille + center: taille - center; 
    y = -taille + center: taille - center;
    
    %% Matrix of u and v
    [v,u] = ndgrid(x,y);
    %% transformation of x and y
    U =   u*cos(phi) - v * sin(phi);  
    V =   u*sin(phi) + v * cos(phi);
    
    
    
    c = w1^2 * U.^2 + w2^2 * V.^2;
    
    f = ((w1*w2) / (2*pi)) * exp(-c/2);
    
    diffw2 = (w1/(2*pi)) * (1 - w2^2*V.^2) .* exp(-c/2);
    
    diffw1 = (w2/(2*pi)) * (1 - w1^2*U.^2) .* exp(-c/2);
    
    sum_psf    = sum(sum(f));
    sum_diffw1 = sum(sum(diffw1));
    sum_diffw2 = sum(sum(diffw2));

end