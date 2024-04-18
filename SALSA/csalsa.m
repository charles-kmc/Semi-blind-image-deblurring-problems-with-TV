% Usage:
% [x_estimate, numA, numAt, objective, distance1, distance2, criterion, ...
% times, mses, ISNR] = csalsa(y, A, mu1, mu2, sigma, varargin)
%
% This function solves the constrained optimization problem 
%
%     arg min   phi( x )
%          x
%       s.t.    || A x - y ||_2 <= epsilon
%
% where A is a generic matrix and phi(.) is a regularizarion 
% function  such that the solution of the denoising problem 
%
%     Psi_tau(y) = arg min_x = 0.5*|| y - x ||_2^2 + tau \phi( x ), 
%
% is known, and the inverse of (alpha I + A^T A) can be computed.
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
%  mu1, mu2 : constraint weights. must be greater than zero.
%       min     Phi(u) + i_epsilon(v)
%       u,v,x
%           s.t.    u = x
%                   v = Ax - y
%           mu1 -> ||x-u||
%           mu2 -> ||Ax-y-v||
%
%   sigma : additive (Gaussian) noise variance
%
%  ===== Optional inputs =============
% 'CONTINUATIONFACTOR' = continuation factor on mu1, mu2.
%           must be greater than 1. default = 1 (no continuation)
%
% 'EPSILON' = epsilon parameter in the inequality ||Ax-y|| < epsilon .
%           default = sqrt(N+sqrt(8*N))*sigma; N = number of elements in y.
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
% 'INVAAT' = term (alpha I + A A^T)^(-1) appearing in the inverse of 
%           (alpha I + A^T A), throught the matrix inversion lemma. Must be a
%           scalar or matrix, to allow multiplication with the result of AT
%           or ATA.
%
% 'INVATA' = function handle to compute the inverse of (alpha I + A A^T), when 
%           the matrix inversion lemma doesn't make a difference; needed to pass updated 
%           values of mu1 and mu2 when using continuation.
%
% 'STOPCRITERION' = type of stopping criterion to use
%                    1 = stop when the criterion ||Ax-y|| < epsilon and relative 
%                        change in the objective function falls below 'ToleranceA'
%                    2 = stop when the criterion ||Ax-y|| < epsilon and relative 
%                        change in the criterion falls below toleranceA
%                    3 = stop when the criterion ||Ax-y|| < epsilon and the
%                    number of iterations has exceeded toleranceA.
%                    Default = 1.
%
%  'TOLERANCEA' = stopping threshold, typically < 1, but a positive integer in case 
%               stopping criterion is 3; Default = 0.001
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
%               min_x phi( W^T(x) ) s.t. ||A x - y || < epsilon
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
%               Must be 1 for analysis prior case.
%
% ===================================================  
% ============ Outputs ==============================
%   x_estimate = solution of the main algorithm
%
%   numA, numAt = number of calls to A and At
%
%   objective = sequence of values of the objective function phi( x )
%
%   distance1, distance2 = sequence of values of the constraint function
%
%   criterion = sequence of values of ||A x - y ||
%
%   times = CPU time after each iteration
%
%   mses = sequence of MSE values, with respect to True_x,
%          if it was given; if it was not given, mses is empty,
%          mses = [].
%
%   ISNR = sequence of values of ISNR, with respect to True_x,
%          if it was given
% ========================================================
function [x_estimate, numA, numAt, objective, distance1, distance2, criterion, times, mses, ISNR] = csalsa(y, A, mu1, mu2, sigma, varargin)

%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 5
     error('Wrong number of required parameters');
end

%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
stopCriterion = 3;
compute_mse = 0;
maxiter = 10000; % outer loop iterations
init = 0;
AT = 0;
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
delta = 1;
epsilon = 0;
compute_ISNR = 0;
numA = 0;
numAt = 0;
isinvATA = 0;
analysis = 0;

%%% check if number of output variables is 10 (means ISNR is requested)
if nargout == 10
    compute_ISNR = 1;
end

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
       %%% check if number of output variables is 10 (means ISNR is requested)
        if nargout == 10 && prod(double((size(true) == size(y))))
            compute_ISNR = 1;
        end
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
        case 'INVATA'
            invATA = varargin{i+1};
            isinvATA = 1;
            
        case 'CONTINUATIONFACTOR'
            delta = varargin{i+1};
        case 'EPSILON'
            epsilon = varargin{i+1};
        case 'ANALYSIS'
            analysis = varargin{i+1};
            %outputvar = 2;
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

%% if the basis W was given, make sure WT is also given, and that their
%% dimesions are compatible.
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
if ~( isa(A, 'function_handle') || isAconvfilter  || isAmask)
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

if ( definedbasis && (~analysis)  )
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
if analysis
    WTATy = WT(ATy);
end

%%%%% verify if ATA is compatible with A and AT;
if (isATA)
    dummy  = ATy;
    ATAd = AT(A(dummy));
    if (ATAd ~= ATA(dummy))
        error('Specified ATA does not seem compatible with the specified A and AT.\n')
    end
end

% if phi was given, check to see if it is a handle and that it 
% accepts two arguments
if exist('psi_function','var')
   if isa(psi_function,'function_handle')
      try  % check if phi can be used, using Aty, which we know has 
           % same size as x
           if analysis
               dummy = psi_function(WTATy,1/mu1);
           else
            dummy = psi_function(ATy,1/mu1); 
           end
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
            if analysis
                dummy = phi_function(WTATy); 
            else
                dummy = phi_function(ATy); 
            end
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

%%%% no SMW lemma used
if (isinvATA)
    if ~isa(invATA, 'function_handle')
        error(['(mu1 I + mu2 AT A)^{-1} must be a function handle.']); 
    end
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
       numAt = numAt + 1;
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
if analysis
    WTx = WT(x);
    z = WTx;
else
    z = x;
end

u = 0*y;
bu = 0*y;
bz = 0*z;

tau = mu1/mu2;

if (~epsilon)
    epsilon = sqrt(numel(y)+8*sqrt(numel(y)))*sigma;
end

t0 = cputime;

if (isAconvfilter)
    H2 = abs(H_FFT).^2;
    filter_FFT = H2./(H2 + tau);
end

if (isAmatrix && (~isinvAAT))
    AAT = matA*matAT;
    invAAT = inv(tau*eye(size(AAT))+AAT);
end

if (isAmask && (~isinvAAT))
    invAAT = 1./(tau+mask);
end

times(1) = 0;
Ax = A(x);
numA = numA + 1;
criterion(1) = norm(Ax(:)-y(:),2);
distance1(1) = norm(Ax(:)-y(:)-u(:),2);

if analysis
    objective(1) = phi_function(z);
    distance2(1) = norm(WTx(:)-z(:),2);
else
    objective(1) = phi_function(x);
    distance2(1) = norm(x(:)-z(:),2);
end

if compute_mse
    if analysis
        Wz = W(z);
        mses(1) = norm(Wz(:)-true(:),2)^2/numel(true);
    else
        Wxtrue = W(true);
        mses(1) = norm(z(:)-true(:),2)^2/numel(true);
    end
        
    if compute_ISNR
        if analysis
            Pnum = norm(true(:)-y(:),2)^2;
            ISNR(1) = 10*log10(Pnum/norm(Wz(:)-true(:),2)^2);
        else
            Wz = W(z);
            Pnum = norm(Wxtrue(:)-y(:),2)^2;
            ISNR(1) = 10*log10(Pnum/norm(Wz(:)-Wxtrue(:),2)^2);
        end
    end
        
end

for outer = 2:maxiter

    mu1inv = 1/mu1;

    for inner = 1:inneriters
        
        if analysis
            r = mu2*AT(y+u+bu)+mu1*W(z-bz);
        else
            r = mu2*AT(y+u+bu)+mu1*(z-bz);
        end
        numAt = numAt + 1;

        if (isAconvfilter)
            if analysis
                x = mu1inv*( r - real( ifft2( filter_FFT.*fft2( r ) ) ) );
            else
                x = mu1inv*( r - WT( real(ifft2(filter_FFT.*fft2( W(r) ))) ) );
            end
        else
            if (isAmatrix)
                    x = mu1inv*(r - AT(invAAT*A(r)));
                    numAt = numAt + 1;
            else
                if (isinvATA)

                    x = invATA(r, mu1, mu2);

                else

                    if (isATA)
                        x = mu1inv*( r -  invAAT*ATA(r) );
                    else
                        if (isAmask)
                            x = mu1inv*(r - invAAT.*A(r));
                        else
                            fprintf('inpainting\n')
                            x = mu1inv*(r - AT(invAAT.*A(r)));
                        end
                    end
                end
            end
        end
        numA = numA + 1;
        
        if analysis
            WTx = WT(x);
            z = psi_function(WTx+bz,mu1inv);
        else
            z = psi_function(x+bz,mu1inv);
        end
    
        Ax = A(x);
        numA = numA + 1;
        ue = Ax-y-bu;
        n_ue = norm(ue(:));
        if n_ue <= epsilon;
            u = ue;
        else
            u =  ue/n_ue*epsilon;
        end
        
    end

    bu = bu-(Ax-y-u);
    
    criterion(outer) = norm(Ax(:)-y(:),2);
    distance1(outer) = norm(Ax(:)-y(:)-u(:),2);
    
    if analysis
        bz = bz-(z-WTx);
        distance2(outer) = norm(WTx(:)-z(:),2);
        objective(outer) = phi_function(z);
    else
        bz = bz-(z-x);
        distance2(outer) = norm(x(:)-z(:),2);
        objective(outer) = phi_function(x);
    end

    if compute_mse
        
        if analysis
            Wz = W(z);
            mses(outer) = norm(Wz(:)-true(:),2)^2/numel(true);
            
            if compute_ISNR
                ISNR(outer) = 10*log10(Pnum/norm(Wz(:)-true(:),2)^2);
            end
            
        else
            mses(outer) = norm(z(:)-true(:),2)^2/numel(true);
            
            if compute_ISNR
                Wz = W(z);
                ISNR(outer) = 10*log10(Pnum/norm(Wz(:)-Wxtrue(:),2)^2);
            end
            
        end
        
    end
    
    times(outer) = cputime - t0;
    
 
    if (verbose)
      fprintf(1,'iter = %d, restriction 1 = %g\t', outer, distance1(outer) )
      fprintf(1,'restriction 2 = %g\t', distance2(outer) )
      fprintf(1,'time = %g seconds\t||Ax-y|| = %g\t', times(outer), criterion(outer) )
      if compute_mse
          fprintf(1,'MSE = %g', mses(outer) )
      end
      fprintf('\n')

    end

    mu1 = mu1*delta;
    mu2 = mu2*delta;
  

    if (outer>1)
        % take no less than miniter and no more than maxiter iterations
        switch stopCriterion
            case 1
                % compute the stopping criterion based on the relative
                % variation of the objective function.
                stopcriterion(outer) = abs(objective(outer)-objective(outer-1))/objective(outer);

                stop_flag = (stopcriterion(outer) < tolA)&&(criterion(outer)<= epsilon);

            case 2
                % compute the stopping criterion based on the relative
                % variation of the estimate.
                stopcriterion(outer) = abs(criterion(outer)-criterion(outer-1))/criterion(outer);
                %stopcriterion(outer) = (criterion(outer-1)-criterion(outer))/criterion(outer-1);
                stop_flag = (stopcriterion(outer) < tolA)&&(criterion(outer)<= epsilon);
            case 3
                % minimum number of iterations

                stop_flag = (outer >= tolA)&&(criterion(outer)<= epsilon);
            otherwise
                error(['Unknown stopping criterion']);
        end

        if stop_flag
            if verbose
                fprintf('Stop criterion satisfied.\n');
            end
            break;
        end

    end
    
end

times(outer) = cputime - t0;

if (outputvar == 2)
    x_estimate = z;
else
    x_estimate = x;
end

