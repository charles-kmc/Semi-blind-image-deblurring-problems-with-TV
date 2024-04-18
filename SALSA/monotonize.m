function y = monotonize(x)

L = length(x);

y(1) = x(1);
offset = 0;

for k = 2:L
    
    if ( x(k) < x(k-1) )
        offset = offset + x(k-1) - x(k);
    end
    
    y(k) = x(k) + offset;
end