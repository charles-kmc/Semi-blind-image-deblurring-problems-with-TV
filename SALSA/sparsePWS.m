function x = sparsePWS(N, L, n)

x = zeros(N,N);

for k = 1:L
    xc = round(rand(2,1)*N);
    
    x( max(xc(1),1):min(xc(1)+n-1,N) , max(xc(2),1):min(xc(2)+n-1,N) ) = 1;
end
