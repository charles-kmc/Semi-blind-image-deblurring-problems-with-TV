%%% This is a modified version of the file deblur_wavelet_FISTA_sep from
%%% the package available at
%%% http://ie.technion.ac.il/~becka/papers/rstls_package.zip

function [x, objective, times, mses] = my_deblur_fista(b,h,tau,Phi_TV, Psi_TV,stopcriterion, tolerance, maxiters, true, verbose)

%L = 2.0506;
L=1;

H_FFT = fft2(h);
HC_FFT = conj(fft2(h));

A = @(x) real(ifft2( H_FFT.*fft2(x) ));
AT = @(x) real(ifft2( HC_FFT.*fft2(x) ));

global calls;
calls = 0;
A = @(x) callcounter(A,x);
AT = @(x) callcounter(AT,x);

x = zeros(size(AT(b)));
y = x;
t = 1;

times(1) = 0;
t0 = cputime;

objective(1) = 0.5*norm(A(x)-b,'fro')^2 + tau*Phi_TV(x);
mses(1) = norm(x-true,'fro')^2/numel(true);
if (verbose)
    fprintf('iter = %d, obj = %3.3g\n', 1, objective(1))
end

for k = 2:maxiters
    x_old = x;
    t_old = t;
    
    y = y-(1/L)*AT(A(y) - b );
    x = Psi_TV(y,tau/L);

    t = 0.5*( 1 + sqrt(1+4*t_old^2) );
    y = x + ( (t_old - 1)/t )*(x - x_old);

    objective(k) = 0.5*norm(A(x)-b,'fro')^2 + tau*Phi_TV(x);
    mses(k) = norm(x-true,'fro')^2/numel(true);

    times(k) = cputime - t0;
           
    switch (stopcriterion)
        case 1
            criterion = abs(objective(k)-objective(k-1))/objective(k);
        case 2
            criterion = norm(x-x_old,'fro')/sqrt(sum(sum( x.^2 )));
        case 3
            criterion = objective(k);
        otherwise
            error('Invalid stopping criterion!');
    end
    
    if (verbose)
        fprintf('iter = %d, obj = %3.3g, stop criterion = %3.3g, ( target = %3.3g )\n', k, objective(k), criterion, tolerance)
    end
    
    if (criterion < tolerance)
        break;
    end
        
end
