%%%% accepts a vectorized image x and returns A(x) with size = M2*N2, if
%%%% mode = 1,
%%%% and returns AT(x) with size = M1*N1, if mode = 2.

function g = A_wrapper(A, AT, x, M1, N1, M2, N2, mode)
if (mode == 1)
    xt = reshape(x, M1, N1);
    gt = A(xt);
    g = reshape(gt, M2*N2, 1);
else if (mode == 2)
        xt = reshape(x, M2, N2);
        gt = AT(xt);
        g = reshape(gt, M1*N1, 1);
    else
        error('The value of parameter mode must be 1 or 2.\n');
    end
end
        