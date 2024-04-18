
function kernel = psf_gaussian(taille, w1, w2, phi)
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
    
    kernel = ((w1*w2) / (2*pi)) * exp(-c/2);
    
    kernel = kernel / sum(sum(kernel));
end