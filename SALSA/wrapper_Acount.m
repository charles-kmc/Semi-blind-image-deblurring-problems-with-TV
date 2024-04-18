function y = wrapper_Acount(A,x)

global Acount;

y = A(x);
Acount = Acount + 1;