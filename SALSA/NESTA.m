%%%% modified version to return a vector containing the CPU time at each
%%%% iteration, starting at 0

% [xk,niter,residuals,outputData] =NESTA(A,At,b,muf,delta,opts)
%
% Solves a L1 minimization problem under a quadratic constraint using the
% Nesterov algorithm, with continuation:
%
%     min_x || U x ||_1 s.t. ||y - Ax||_2 <= delta
% 
% Continuation is performed by sequentially applying Nesterov's algorithm
% with a decreasing sequence of values of  mu0 >= mu >= muf
%
% The primal prox-function is also adapted by accounting for a first guess
% xplug that also tends towards x_muf 
%
% The observation matrix A is a projector
%
% Inputs:   A and At - measurement matrices (either a matrix, in which
%               case At is unused, or function handles)
%           b   - Observed ata, a m x 1 array
%           muf - The desired value of mu at the last continuation step.
%               A smaller mu leads to higher accuracy.
%           delta - l2 error bound.  This enforces how close the variable
%               must fit the observations b, i.e. || y - Ax ||_2 <= delta
%               If delta = 0, enforces y = Ax
%           opts -
%               This is a structure that contains additional options,
%               some of which are optional.
%               The fieldnames are case insensitive.  Below
%               are the possible fieldnames:
%               
%               opts.xplug - the first guess for the primal prox-function, and
%                 also the initial point for xk.  By default, xplug = At(b)
%
%               opts.U and opts.Ut - Anaysis/Synthesis operators
%                 (either matrices of function handles).
%
%               opts.MaxIntIter - number of continuation steps.
%                 default is 5
%
%               opts.maxiter - max number of iterations in an inner loop.
%                 default is 10,000
%
%               opts.TolVar - tolerance for the stopping criteria
%
%               opts.stopTest - which stopping criteria to apply
%                   opts.stopTest == 1 : stop when the relative
%                       change in the objective function is less than
%                       TolVar
%                   opts.stopTest == 2 : stop with the l_infinity norm
%                       of difference in the xk variable is less
%                       than TolVar
%
%               opts.TypeMin - if this is 'L1' (default), then
%                   minimizes a smoothed version of the l_1 norm.
%                   If this is 'tv', then minimizes a smoothed
%                   version of the total-variation norm.
%                   The string is case insensitive.
%
%               opts.Verbose - if this is 0 or false, then very
%                   little output is displayed.  If this is 1 or true,
%                   then output every iteration is displayed.
%                   If this is a number p greater than 1, then
%                   output is displayed every pth iteration.
%
%               opts.fid - if this is 1 (default), the display is
%                   the usual Matlab screen.  If this is the file-id
%                   of a file opened with fopen, then the display
%                   will be redirected to this file.
%
%               opts.errFcn - if this is a function handle,
%                   then the program will evaluate opts.errFcn(xk)
%                   at every iteration and display the result.
%                   ex.  opts.errFcn = @(x) norm( x - x_true )
%
%               opts.outFcn - if this is a function handle, 
%                   then then program will evaluate opts.outFcn(xk)
%                   at every iteration and save the results in outputData.
%                   If the result is a vector (as opposed to a scalar),
%                   it should be a row vector and not a column vector.
%                   ex. opts.outFcn = @(x) [norm( x - xtrue, 'inf' ),...
%                                           norm( x - xtrue) / norm(xtrue)]
%  Outputs:
%           xk  - estimate of the solution x
%
%           niter - number of iterations
%
%           residuals - first column is the residual at every step,
%               second column is the value of f_mu at every step
%
%           outputData - a matrix, where each row r is the output
%               from opts.outFcn, if supplied.
%       
%
% Written by: Jerome Bobin, Caltech
% Email: bobin@acm.caltech.edu
% Created: February 2009
% Modified: May 2009, Jerome Bobin and Stephen Becker, Caltech
%
% Version 1.0
%   See also Core_Nesterov


function [xk,niter,residuals,outputData, times] =NESTA(A,At,b,muf,delta,opts)

if nargin < 6, opts = []; end
if isempty(opts) && isnumeric(opts), opts = struct; end

%---- Set defaults
fid = setOpts('fid',1);
Verbose = setOpts('Verbose',true);
function printf(varargin), fprintf(fid,varargin{:}); end
MaxIntIter = setOpts('MaxIntIter',5,1);
TypeMin = setOpts('TypeMin','L1');
TolVar = setOpts('tolvar',1e-5);
U = setOpts('U', @(x) x );
Ut = setOpts('Ut', @(x) x );
xplug = setOpts('xplug',[]);

residuals = []; outputData = [];

if isempty(xplug)
    if isa(A,'function_handle')
        xplug=At(b);
    else
        xplug = A.'*b;
    end
end

if norm(xplug) < 1e-12
     if isa(A,'function_handle')
         x_ref=At(b);
     else
         x_ref = A.'*b;
     end
else
    x_ref = xplug;
end

switch TypeMin
    case 'L1'
        mu0 = 0.9*max(abs(U(x_ref)));
    case 'tv'
        mu0 = ValMUTv(U(x_ref));
end

niter = 0;
Gamma = (muf/mu0)^(1/MaxIntIter);
mu = mu0;
Gammat= (TolVar/0.1)^(1/MaxIntIter);
TolVar = 0.1;
 
times = [];
t0 = cputime;

for nl=1:MaxIntIter
    
    mu = mu*Gamma;
    TolVar=TolVar*Gammat;    opts.TolVar = TolVar;
    opts.xplug = xplug;
    if Verbose, printf('\tBeginning %s Minimization; mu = %g\n',opts.TypeMin,mu); end
    [xk,niter_int,res,out, times_nc] = Core_Nesterov(...
        A,At,b,mu,delta,opts);
    times = [times; times_nc'];
    
    xplug = xk;
    niter = niter_int + niter;
    
    residuals = [residuals; res];
    outputData = [outputData; out];

end

times(end) = cputime - t0;

%---- internal routine for setting defaults
function var = setOpts(field,default,mn,mx)
    var = default;
    % has the option already been set?
    if ~isfield(opts,field) 
        % see if there is a capitalization problem:
        names = fieldnames(opts);
        for i = 1:length(names)
            if strcmpi(names{i},field)
                opts.(field) = opts.(names{i});
                rmfield(opts,names{i});
                break;
            end
        end
    end
    if isfield(opts,field) && ~isempty(opts.(field))
        var = opts.(field);  % override the default
    end
    % perform error checking, if desired
    if nargin >= 3 && ~isempty(mn)
        if var < mn
            printf('Variable %s is %f, should be at least %f\n',...
                field,var,mn); error('variable out-of-bounds');
        end
    end
    if nargin >= 4 && ~isempty(mx)
        if var > mx
            printf('Variable %s is %f, should be at least %f\n',...
                field,var,mn); error('variable out-of-bounds');
        end
    end
    opts.(field) = var;
end




%---- internal routine for setting mu0 in the tv minimization case
function th=ValMUTv(x);

    N = length(x);n = floor(sqrt(N));
    Dv = spdiags([reshape([-ones(n-1,n); zeros(1,n)],N,1) ...
            reshape([zeros(1,n); ones(n-1,n)],N,1)], [0 1], N, N);
        Dh = spdiags([reshape([-ones(n,n-1) zeros(n,1)],N,1) ...
            reshape([zeros(n,1) ones(n,n-1)],N,1)], [0 n], N, N);
        D = sparse([Dh;Dv]);


    Dhx = Dh*x;
    Dvx = Dv*x;
    
    sk = sqrt(abs(Dhx).^2 + abs(Dvx).^2);
    th = max(sk);

end

end %-- end of NESTA function
