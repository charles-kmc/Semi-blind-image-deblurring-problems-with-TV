% File: SALSA.m
%
% Usage:
% [x_estimate, objective, distance, times, mses] = SALSA(y,A,tau,varargin)
%
% This function solves the regularization problem 
%
%     arg min_x = 0.5*|| y - A x ||_2^2 + tau phi( x ), 
%
% where A is a generic matrix and phi(.) is a regularizarion 
% function  such that the solution of the denoising problem 
%
%     Psi_tau(y) = arg min_x = 0.5*|| y - x ||_2^2 + tau \phi( x ), 
%
% is known. 
% 
% -----------------------------------------------------------------------
%
% For further details about the SALSA and C-SALSA algorithms, see the papers: 
%
% [1] M. Afonso, J. Bioucas-Dias and M. Figueiredo. "Fast Algorithm for the Constrained
% Formulation of Compressive Image Reconstruction and Other Linear Inverse Problems",
% Submitted to IEEE International Conf. on Acoustics, Speech, and Signal Processing
% (ICASSP), Dallas, U.S.A., 2010. 
% http://arxiv.org/abs/0909.3947v1
%
% [2] M. Afonso, J. Bioucas-Dias and M. Figueiredo. "Fast Image Recovery Using Variable
% Splitting and Constrained Optimization", IEEE Transactions on Image Processing,
% (Submitted), 2009.
% http://arxiv.org/abs/0910.4887
%
% [3] M. Figueiredo and J. Bioucas-Dias and M. Afonso. "Fast Frame-Based Image Decon-
% volution Using Variable Splitting and Constrained Optimization", IEEE Worskhop on
% Statistical Signal Processing, Cardiff, Wales, 2009.
% http://arxiv.org/abs/0904.4872
%
% -----------------------------------------------------------------------
% Authors: Manya Afonso, Jose Bioucas-Dias and Mario Figueiredo
% Date: October, 2009.
% 
% For any technical queries/comments/bug reports, please contact:
% manya (dot) afonso (at) gmail (dot) com
% -----------------------------------------------------------------------
%
% Copyright (2009): Manya Afonso, Jose Bioucas-Dias, and Mario Figueiredo
% 
% SALSA is distributed under the terms of 
% the GNU General Public License 2.0.
% 
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
% 
%  ===== Required inputs =============
%
%  y: 1D vector or 2D array (image) of observations
%     
%  A: if y and x are both 1D vectors, A can be a 
%     k*n (where k is the size of y and n the size of x)
%     matrix or a handle to a function that computes
%     products of the form A*v, for some vector v.
%     In any other case (if y and/or x are 2D arrays), 
%     A can be passed as a handle to a function which computes 
%     products of the form A*x; another handle to a function 
%     AT which computes products of the form A'*y is also required 
%     in this case. A can also be the convolution kernel, or mask.
%     The size of x is determined as the size
%     of the result of applying AT.
%
%  tau: regularization parameter, usually a non-negative real 
%       parameter of the objective  function (see above). 
%  
%
%  ===== Optional inputs =============
%  
% 'MU' = parameter mu (weight for constraint function as a result of
%        splitting); default = 1e-3;
%
%  'Psi' = denoising function handle; handle to denoising function
%          Default = soft threshold.
%
%  'Phi' = function handle to regularizer needed to compute the objective
%          function.
%          Default = ||x||_1
%
%  'AT'  = function handle for the function that implements
%          the multiplication by the conjugate of A, when A
%          is a function handle. 
%          If A is an array, AT is ignored.
%
% 'BASIS' = Synthesis operator (inverse wavelet transform, in case of wavelet frames); 
%           default = identity
% 
% 'BASISTRANSPOSE' = Analysis operator (forward transform); default = identity
% 
% 'MASK' = must be set to 1, to indicate that A is not a masking
%           operator, but the mask (in the spatial domain). Do not specify
%           AT or INVAAT or ATA when using this.
%
% 'CONVOLUTIONFILTER' = must be set to 1, to indicate that A is not a
%           convolution operator, but the kernel (of the same size as the image). 
%           Do not specify AT or INVAAT or ATA when using this.
%
% 'ATA' = function handle or matrix for AT(A(x)). May be needed when specified A
%           and AT are operators, especially when the observation is in
%           some transform domain. Must be used in combination with INVAAT.
%
% 'INVAAT' = term (mu I + A A^T)^(-1) appearing in the inverse of 
%           (mu I + A^T A), throught the matrix inversion lemma. Must be a
%           scalar or matrix, to allow multiplication with the result of AT
%           or ATA.
%
% 'STOPCRITERION' = type of stopping criterion to use
%                    1 = stop when the relative 
%                        change in the objective function 
%                        falls below 'ToleranceA'
%                    2 = stop when the relative norm of the difference between 
%                        two consecutive estimates falls below toleranceA
%                    3 = stop when the objective function 
%                        becomes equal or less than toleranceA.
%                    Default = 1.
%
%  'TOLERANCEA' = stopping threshold; Default = 0.001
% 
%  'MAXITERA' = maximum number of iterations allowed in the
%               main phase of the algorithm.
%               Default = 10000
%
%  'INNERITERS' = number of inner loop iterations; default = 1
%
%  'INITIALIZATION' must be one of {0,1,2,array}
%               0 -> Initialization at zero. 
%               1 -> Random initialization.
%               2 -> initialization with A'*y.
%               array -> initialization provided by the user.
%               Default = 0;
%
%  'TRUE_X' = if the true underlying x is passed in 
%                this argument, MSE evolution is computed
%
%  'ANALYSIS' if non-zero and wavelet basis specified, solves 
%               min_x 0.5||A x - y ||^2 + tau *phi( W^T(x) )
%               default = 0 (synthesis prior)
%
%  'VERBOSE'  = work silently (0) or verbosely (1), default = 1
%
% 'OUTPUTVARIABLE' = which of the variables to return
%               min 0.5 ||Ax -y ||^2 + tau phi(z)
%               x,z
%                   s.t. x = z
%
%               1 (default); returns x
%               2 returns z
%
%
% ===================================================  
% ============ Outputs ==============================
%   x_estimate = solution of the main algorithm
%
%   objective = sequence of values of the objective function  
%               0.5*|| y - A x ||_2^2 + tau phi( x )
%
%   distance = sequence of values of the constraint function
%
%   times = CPU time after each iteration
%
%   mses = sequence of MSE values, with respect to True_x,
%          if it was given; if it was not given, mses is empty,
%          mses = [].
% ========================================================

function [x_estimate, objective, distance, times, mses] = ...
         SALSA(y,A,tau,varargin)

%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 3
     error('Wrong number of required parameters');
end

%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
stopCriterion = 1;
compute_mse = 0;
maxiter = 10000; % outer loop iterations
init = 0;
AT = 0;
mu = 1e-3;
inneriters = 1;
psi_ok = 0;
tolA = 0.001;
isAconvfilter = 0;
isAmatrix = 0;
isAmask = 0;
verbose = 1;
definedbasis = 0;
definedbasistranspose = 0;
outputvar = 1;
isAT = 0;
isATA = 0;
isinvAAT = 0;

%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
  error('Optional parameters should always go by pairs');
else
  for i=1:2:(length(varargin)-1)
    switch upper(varargin{i})
     case 'BASIS'
         definedbasis = 1;
       W = varargin{i+1};   
     case 'BASISTRANSPOSE'
         definedbasistranspose = 1;
       WT = varargin{i+1};
     case 'MASK'
         isAmask = varargin{i+1};   
     case 'CONVOLUTIONFILTER'
       isAconvfilter = varargin{i+1};
     case 'PSI'
       psi_function = varargin{i+1};
     case 'PHI'
       phi_function = varargin{i+1};
     case 'MU'
       mu = varargin{i+1};
     case 'STOPCRITERION'
       stopCriterion = varargin{i+1};
     case 'TOLERANCEA'       
       tolA = varargin{i+1};
     case 'INNERITERS'
         inneriters = varargin{i+1};
     case 'MAXITERA'
       maxiter = varargin{i+1};
     case 'INITIALIZATION'
       if numel(varargin{i+1}) > 1   % we have an initial x
	 init = 33333;    % some flag to be used below
	 x = varargin{i+1};
       else 
	 init = varargin{i+1};
       end
     case 'TRUE_X'
       compute_mse = 1;
       true = varargin{i+1};
     case 'AT'
         isAT = 1;
       AT = varargin{i+1};
     case 'VERBOSE'
         verbose = varargin{i+1};
     case 'OUTPUTVARIABLE'
         outputvar = varargin{i+1};
     case 'ATA'
         ATA = varargin{i+1};
         isATA = 1;
     case 'INVAAT'
         invAAT = varargin{i+1};
         isinvAAT = 1;
     otherwise
      % Hmmm, something wrong with the parameter string
      error(['Unrecognized option: ''' varargin{i} '''']);
    end;
  end;
end
%%%%%%%%%%%%%%

if (sum(stopCriterion == [1 2 3])==0)
   error(['Unknown stopping criterion']);
end

%%%% if the basis W was given, make sure WT is also given, and that their
%%%% dimensions are compatible.
if xor( definedbasis,definedbasistranspose )
    error(['If you give W you must also give WT, and vice versa.']);
end

if (~definedbasis)
    W = @(x) x;
    WT = @(x) x;
end


% if A is a function handle, we have to check presence of AT,
if isa(A, 'function_handle') && ~isa(AT,'function_handle')
   error(['The function handle for transpose of A is missing']);
end 

% if A is a convolution filter, we have to generate handles to the
% operators A and AT
if (isAconvfilter)
    h = A;
    H_FFT = fft2(h);
    HC_FFT = conj(H_FFT); 
    
    B = @(x) real(ifft2(H_FFT.*fft2(x)));
    BT = @(x) real(ifft2(HC_FFT.*fft2(x)));
    
end

% if A is a mask, we need to generate the handles for A and AT
if (isAmask)
    mask = A;
    B = @(x) mask.*x;
    BT = @(x) mask.*x;
end

% if A is a matrix, we find out dimensions of y and x,
% and create function handles for multiplication by A and A',
% so that the code below doesn't have to distinguish between
% the handle/not-handle cases
if ~( isa(A, 'function_handle') || isAconvfilter  || isAmask )
    isAmatrix = 1;
    
    if ~(size(y,1) == size(A,1) && size(y,2) == 1 )
        error('For a MxN observation matrix A, the measurement vector y must be a length M vector.');
    end
    
    if compute_mse && ~(size(true,1) == size(A,2) && size(true,2) == 1 )
        fprintf('\nWARNING: For a MxN observation matrix A, the vector x_true must be a length N vector.\n');
        compute_mse = 0;
    end
    matA = A;
    matAT = A';
   BT = @(x) matAT*x;
   B = @(x) matA*x;
end

if ( definedbasis )
        A = @(x) B(W(x));
        AT = @(x) WT(BT(x));
else
    if (~isAT)
        A = B;
        AT = BT;
    end
end
% from this point down, A and AT are always function handles.

% Precompute A'*y since it'll be used a lot
ATy = AT(y);

%%%%% verify if ATA is compatible with A and AT;
if (isATA)
    dummy  = ATy;
    ATAd = AT(A(dummy));
    if (ATAd ~= ATA(dummy))
        error('Specified ATA does not seem compatible with the specified A and AT.\n')
    end
end

%%%% verify if invAAT has dimensions compatible with A and AT
if(isinvAAT)
    if isa(invAAT, 'function_handle')
        error('Specified invAAT must be a scalar or a matrix of compatible dimensions.\n')
    else
        if (numel(invAAT)>1)
            dummy = ATy;
            dummy_prod = AT(invAAT*A(dummy));
            if ~((size(dummy,1)==size(dummy_prod,1))&&(size(dummy,2)==size(dummy_prod,2)))
                error('Specified invAAT does not seem compatible with the specified A and AT.\n')
            end
        end
    end
end

% if phi was given, check to see if it is a handle and that it 
% accepts two arguments
if exist('psi_function','var')
   if isa(psi_function,'function_handle')
      try  % check if phi can be used, using Aty, which we know has 
           % same size as x
            dummy = psi_function(ATy,tau); 
            psi_ok = 1;
      catch
         error(['Something is wrong with function handle for psi'])
      end
   else
      error(['Psi does not seem to be a valid function handle']);
   end
else %if nothing was given, use soft thresholding
   psi_function = @(x,tau) soft(x,tau);
end

% if psi exists, phi must also exist
if (psi_ok == 1)
   if exist('phi_function','var')
      if isa(phi_function,'function_handle')
         try  % check if phi can be used, using Aty, which we know has 
              % same size as x
              dummy = phi_function(ATy); 
         catch
           error(['Something is wrong with function handle for phi'])
         end
      else
        error(['Phi does not seem to be a valid function handle']);
      end
   else
      error(['If you give Psi you must also give Phi']); 
   end
else  % if no psi and phi were given, simply use the l1 norm.
   phi_function = @(x) sum(abs(x(:))); 
end

%--------------------------------------------------------------
% Initialization
%--------------------------------------------------------------
switch init
    case 0   % initialize at zero, using AT to find the size of x
       x = AT(zeros(size(y)));
    case 1   % initialize randomly, using AT to find the size of x
       x = randn(size(AT(zeros(size(y)))));
    case 2   % initialize x0 = A'*y
       x = ATy; 
    case 33333
       % initial x was given as a function argument; just check size
       if size(A(x)) ~= size(y)
          error(['Size of initial x is not compatible with A']); 
       end
    otherwise
       error(['Unknown ''Initialization'' option']);
end

% if the true x was given, check its size
if compute_mse & (size(true) ~= size(x))  
   error(['Initial x has incompatible size']); 
end

% initializing
z = AT(zeros(size(y)));
b = z;
muinv = 1/mu;
threshold = tau/mu;
criterion(1) = 1;

% Compute and store initial value of the objective function
resid =  y-A(x);
prev_f = 0.5*(resid(:)'*resid(:)) + tau*phi_function(x);

if verbose
    fprintf('Initial value of objective function = %3.3g\n',prev_f)
end

% start the clock
t0 = cputime;

if (isAconvfilter)
    filter_FFT = HC_FFT./(abs(H_FFT).^2 + mu).*H_FFT;
end

if (isAmatrix && (~isinvAAT))
    AAT = matA*matAT;
    invAAT = inv(mu*eye(size(AAT))+AAT);
end

if (isAmask && (~isinvAAT))
    invAAT = 1./(mu+mask);
end

times(1) = 0;
objective(1) = prev_f;

if compute_mse
   mses(1) = sum(sum((x-true).^2))/numel(x);
end

global numA;

for outer = 1:maxiter
      
    for inner = 1:inneriters
        
        z = psi_function(x-b, threshold);
        r = ATy + mu*(z+b);
        
        if (isAconvfilter)
            x = muinv*( r - WT( real(ifft2(filter_FFT.*fft2( W(r) ))) ) );
            numA = numA + 1;
        else
            if (isATA)
                x = muinv*( r -  invAAT*ATA(r) );
            else
                if (isAmatrix)
                    x = muinv*(r - AT(invAAT*A(r)));
                else
                    if (isAmask)
                        x = muinv*(r - AT(invAAT.*A(r)));
                    end
                end
            end
        end
        
    end
    
    b = b + (z - x);
    
    resid =  y-A(x);
    objective(outer+1) = 0.5*(resid(:)'*resid(:)) + tau*phi_function(x);
    
    if compute_mse
        err = x-true;
        mses(outer+1) =  (err(:)'*err(:))/numel(x);
    end
    
    distance(outer) = norm(x(:)-z(:),2)/sqrt(norm(x(:),2)^2 + norm(z(:),2)^2);
    
    if (outer>1)
        % take no more than maxiter iterations
        switch stopCriterion
            case 1
                % compute the stopping criterion based on the relative
                % variation of the objective function.
                criterion(outer) = abs(objective(outer+1)-objective(outer))/objective(outer);
            case 2
                % compute the stopping criterion based on the relative
                % variation of the estimate.
                criterion(outer) = abs(norm(x(:)-xprev(:))/norm(x(:)));
            case 3
                % continue if not yet reached target value tolA
                criterion(outer) = objective(outer+1);
            otherwise
                error(['Unknown stopping criterion']);
        end
        
         if ( criterion(outer) < tolA )
             if verbose
                fprintf('\niter = %d, obj = %3.3g, ||x-z|| = %3.3g, stop criterion = %3.3g, ( target = %3.3g )\t', outer, objective(outer+1), distance(outer), criterion(outer), tolA)
                if compute_mse
                    fprintf('MSE = %g',mses(outer+1))
                end

                fprintf('\nConvergence reached for Bregman iterative procedure.\n')
             end
             times(outer+1) = cputime - t0;
             break;
         end
         
    end
    
    if verbose
        fprintf('\niter = %d, obj = %3.3g, ||x-z|| = %3.3g, stop criterion = %3.3g, ( target = %3.3g )\t', outer, objective(outer+1), distance(outer), criterion(outer), tolA)
        if compute_mse
            fprintf('MSE = %g',mses(outer+1))
        end
    end
    
    times(outer+1) = cputime - t0;
end

times(outer) = cputime - t0;

if (outputvar == 2)
    x_estimate = z;
else
    x_estimate = x;
end