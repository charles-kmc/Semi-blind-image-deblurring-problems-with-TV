function y = wrapper_Atcount(A,x)

global Atcount;

y = A(x);
Atcount = Atcount + 1;