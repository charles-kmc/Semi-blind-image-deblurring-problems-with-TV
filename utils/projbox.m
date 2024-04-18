function [x] = projbox(x, min_x, max_x)
    x = min(max(x, min_x), max_x);
end