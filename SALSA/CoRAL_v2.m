%function [x_hat, objective, times, mses, ISNRs] = ADMM_compoundregularization(y, h_kernel, true_x, tau1, Phi1, Psi1, mu1, tau2, Phi2, Psi2, mu2, maxiters, tol)
function [x, numA, numAt, objective, distance, times, mses] = ...
         CoRAL(y,A,tau1,tau2,varargin)

%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 4
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

mu1 = 1e-3;
mu2 = 1e-3;

inneriters = 1;
psi1_ok = 0;
psi2_ok = 0;
tolA = 0.001;
verbose = 1;

isLS = 0;
numA = 0;
numAt = 0;

definedP1 = 0;
definedP1T = 0;

definedP2 = 0;
definedP2T = 0;

isTVinitialization1 = 0;
TViters1 = 5;

isTVinitialization2 = 0;
TViters2 = 5;

isinvLS = 0;
%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
  error('Optional parameters should always go by pairs');
else
  for i=1:2:(length(varargin)-1)
    switch upper(varargin{i})
     case 'W'
         definedW = 1;
       W = varargin{i+1};   
     case 'WT'
         definedWT = 1;
       WT = varargin{i+1};
     case 'P1'
         definedP1 = 1;
       P1 = varargin{i+1};   
     case 'P1T'
         definedP1T = 1;
       P1T = varargin{i+1};  
     case 'P2'
         definedP2 = 1;
       P2 = varargin{i+1};   
     case 'P2T'
         definedP2T = 1;
       P2T = varargin{i+1};  
     case 'MASK'
         isAmask = varargin{i+1};   
     case 'UNITARYTRANSFORMDOMAINMASK'
       isUnitaryTransformDomainMask = varargin{i+1};
     case 'CONVOLUTIONFILTER'
       isAconvfilter = varargin{i+1};
     case 'PSI1'
       psi1 = varargin{i+1};
     case 'PHI1'
       phi1 = varargin{i+1};
     case 'TVINITIALIZATION1'
        isTVinitialization1 = varargin{i+1};
     case 'TVITERS1'
        TViters1 = varargin{i+1};       
    case 'PSI2'
       psi2 = varargin{i+1};
     case 'PHI2'
       phi2 = varargin{i+1};
     case 'TVINITIALIZATION2'
        isTVinitialization2 = varargin{i+1};
     case 'TVITERS2'
        TViters2 = varargin{i+1};       
	 case 'MU1'
       mu1 = varargin{i+1};
     case 'MU2'
       mu2 = varargin{i+1};
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
     case 'LS'
         invLS = varargin{i+1};
         isinvLS = 1;
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

mu = mu1 + mu2;
muinv = 1/mu;

%%%% if the analysis operator P1 was given, make sure P1T is also given, and that their
%%%% dimensions are compatible.
if xor( definedP1,definedP1T )
    error(['If you give P1 you must also give P1T, and vice versa.']);
end

if (~definedP1)
    P1 = @(x) x;
    P1T = @(x) x;
end

%%%% same check for P2, P2T
if xor( definedP2,definedP2T )
    error(['If you give P2 you must also give P2T, and vice versa.']);
end

if (~definedP2)
    P2 = @(x) x;
    P2T = @(x) x;
end

% if A is a function handle, we have to check presence of AT,
if isa(A, 'function_handle') && ~isa(AT,'function_handle')
   error(['The function handle for transpose of A is missing']);
end 

% if A is a matrix, we find out dimensions of y and x,
% and create function handles for multiplication by A and A',
% so that the code below doesn't have to distinguish between
% the handle/not-handle cases
if ~( isa(A, 'function_handle') )
    
    if ~(size(y,1) == size(A,1) && size(y,2) == 1 )
        error('For a MxN observation matrix A, the measurement vector y must be a length M vector.');
    end
    
    if compute_mse && ~(size(true,1) == size(A,2) && size(true,2) == 1 )
        fprintf('\nWARNING: For a MxN observation matrix A, the vector x_true must be a length N vector.\n');
        compute_mse = 0;
    end
    matA = A;
    matAT = A';
   AT = @(x) matAT*x;
   A = @(x) matA*x;
end

% from this point down, A and AT are always function handles.

% Precompute A'*y since it'll be used a lot
ATy = AT(y);
numAt = numAt + 1;


% if A is a function handle, we have to check presence of invLS,
if isa(A, 'function_handle')
    
    if ( ~isinvLS || ~isa(invLS, 'function_handle'))
        error(['(A^T A + \mu I)^(-1) must be specified as a function handle.\n']);
    else
        dummy = invLS(ATy);
        if ~((size(dummy,1)==size(ATy,1))&&(size(dummy,2)==size(ATy,2)))
            error('Specified function handle for solving the LS step does not seem compatible with the specified A and AT.\n')
        end
    end
else %% A is a matrix
    [M,N] = size(matA);
    if ( M > N )
        ATA = transpose(matA)*matA;
        inverse_term = inv(ATA+mu*eye(N,N));
        %inverse_term = @(x) invATA*x;
    else
        AAT = matA*transpose(matA);
        inverse_term = AT*inv(mu*eye(M,M)+AAT)*A;
    end
    invLS = @(x) inverse_term*x;        
end

% from this point down, A and AT are always function handles.
% Precompute A'*y since it'll be used a lot
ATy = AT(y);
numAt = numAt + 1;


if isTVinitialization1 && isTVinitialization2
    fprintf('WARNING: TV with initialization has been specified for both Phi1 and Phi2. Try reformulating with a single regularizer term for efficiency.\n')
end

% if psi1 was given, check to see if it is a handle and that it 
% accepts two arguments
if exist('psi1','var')
    if isTVinitialization1
        fprintf('WARNING: user specified Phi1 and Psi1 will not be used as TV with initialization flag has been set to 1.\n')
    else
       if isa(psi1,'function_handle')
          try  % check if phi1 can be used, using Aty, which we know has 
               % same size as x
               dummy = psi1(P1T(ATy),tau1); 
               psi1_ok = 1;
          catch
             error(['Something is wrong with function handle for psi1'])
          end
       else
          error(['Psi1 does not seem to be a valid function handle']);
       end
    end
else %if nothing was given, use soft thresholding
   psi1 = @(x,tau) soft(x,tau);
end

% if psi1 exists, phi1 must also exist
if (psi1_ok == 1)
   if exist('phi1','var')
      if isa(phi1,'function_handle')
         try  % check if phi1 can be used, using Aty, which we know has 
              % same size as x
              dummy = phi1( P1(ATy) ); 
         catch
           error(['Something is wrong with function handle for phi1'])
         end
      else
        error(['Phi1 does not seem to be a valid function handle']);
      end
   else
      error(['If you give Psi1 you must also give Phi1']); 
   end
else  
   if ~isTVinitialization1
       phi1 = @(x) sum(abs(x(:))); 
   else
       % if no psi1 and phi1 were given, simply use the l1 norm.
       phi1 = @(x) TVnorm(x);
   end
end


% if psi2 was given, check to see if it is a handle and that it 
% accepts two arguments
if exist('psi2','var')
    if isTVinitialization2
        fprintf('WARNING: user specified Phi2 and Psi2 will not be used as TV with initialization flag has been set to 1.\n')
    else
       if isa(psi2,'function_handle')
          try  % check if phi can be used, using Aty, which we know has 
               % same size as x
               dummy = psi2(P2T(ATy),tau2); 
               psi2_ok = 1;
          catch
             error(['Something is wrong with function handle for psi2'])
          end
       else
          error(['Psi2 does not seem to be a valid function handle']);
       end
    end
else %if nothing was given, use soft thresholding
   psi2 = @(x,tau) soft(x,tau);
end

% if psi2 exists, phi2 must also exist
if (psi2_ok == 1)
   if exist('phi2','var')
      if isa(phi2,'function_handle')
         try  % check if phi2 can be used, using Aty, which we know has 
              % same size as x
              dummy = phi2( P2(ATy) ); 
         catch
           error(['Something is wrong with function handle for phi2'])
         end
      else
        error(['Phi2 does not seem to be a valid function handle']);
      end
   else
      error(['If you give Psi2 you must also give Phi2']); 
   end
else  
   if ~isTVinitialization2
       phi2 = @(x) sum(abs(x(:))); 
   else
       % if no psi2 and phi2 were given, simply use the l1 norm.
       phi2 = @(x) TVnorm(x);
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


P1Tx = P1T(x);
P2Tx = P2T(x);

% initializing
u = P1Tx;
bu = u;
mu1inv = 1/mu1;
threshold1 = tau1/mu1;

v = P2Tx;
bv = v;
mu2inv = 1/mu2;
threshold2 = tau2/mu2;

criterion(1) = 1;

% Compute and store initial value of the objective function
resid =  y-A(x);
numA = numA + 1;
prev_f = 0.5*(resid(:)'*resid(:)) + tau1*phi1(u) + tau2*phi2(v);

if verbose
    fprintf('Initial value of objective function = %3.3g\n',prev_f)
end

% start the clock
t0 = cputime;

times(1) = 0;
objective(1) = prev_f;

if compute_mse
   mses(1) = sum(sum((x-true).^2))/numel(x);
end

if isTVinitialization1
    pux = 0*u;
    puy = 0*u;
end

if isTVinitialization2
    pvx = 0*v;
    pvy = 0*v;
end

for outer = 1:maxiter
    
    xprev = x;
      
    %for inner = 1:inneriters
        
        if isTVinitialization1
               [u,pux,puy] = chambolle_prox_TV_stop(real(P1Tx-bu), 'lambda', threshold1, 'maxiter', TViters1, 'dualvars',[pux puy]);
        else
            u = psi1(P1Tx-bu, threshold1);
        end
        if isTVinitialization2
               [v,pvx,pvy] = chambolle_prox_TV_stop(real(P2Tx-bv), 'lambda', threshold2, 'maxiter', TViters2, 'dualvars',[pvx pvy]);
        else
            v = psi2(P2Tx-bv, threshold2);
        end
        
        r = ATy + mu1*P1(u+bu) + mu2*P2(v+bv);
                
        x = invLS(r);
        
    %end
    
    P1Tx = P1T(x);
    P2Tx = P2T(x);
    
    bu = bu + (u - P1Tx);
    bv = bv + (v - P2Tx);
    
    resid =  y-A(x);
    numA = numA + 1;
    objective(outer+1) = 0.5*(resid(:)'*resid(:)) + tau1*phi1(u) + tau2*phi2(v);
    
    if compute_mse
        err = x-true;
        mses(outer+1) =  (err(:)'*err(:))/numel(x);
    end
    
    distance(outer,1) = norm(P1Tx(:)-u(:),2)/sqrt(norm(P1Tx(:),2)^2 + norm(u(:),2)^2);
    distance(outer,2) = norm(P2Tx(:)-v(:),2)/sqrt(norm(P2Tx(:),2)^2 + norm(v(:),2)^2);
    
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
                fprintf('\niter = %d, obj = %3.3g, stop criterion = %3.3g, ( target = %3.3g )\t', outer, objective(outer+1), criterion(outer), tolA)
                if compute_mse
                    fprintf('MSE = %g',mses(outer+1))
                end

                fprintf('\nConvergence reached.\n')
             end
             times(outer+1) = cputime - t0;
             break;
         end
         
    end
    
    if verbose
        fprintf('\niter = %d, obj = %3.3g, stop criterion = %3.3g, ( target = %3.3g )\t', outer, objective(outer+1), criterion(outer), tolA)
        if compute_mse
            fprintf('MSE = %g',mses(outer+1))
        end
    end
    
    times(outer+1) = cputime - t0;
end


% if (isAconvfilter)
%     if ~definedW
%         filter_FFT = 1./(abs(H_FFT).^2 + alpha);
%     else
%         filter_FFT = HC_FFT./(abs(H_FFT).^2 + alpha).*H_FFT;
%     end
% end
% 
% if isAmatrix %&& (~isinvAAT)
%     AAT = matA*matAT;
%     invAAT = matAT*inv(alpha*eye(size(AAT))+AAT)*matA;
% end
% 
% if (isAmask)% && (~isinvAAT))
%     invAAT = (1/(alpha+1));%1./(alpha+mask);
% end
% 
% times(1) = 0;
% objective(1) = prev_f;
% 
% if compute_mse
%    mses(1) = sum(sum((x-true).^2))/numel(x);
% end
% 
% for outer = 1:maxiter
%       
%     for inner = 1:inneriters
%         
%         u = psi1(P1Tx-bu, threshold1);
%         v = psi2(P2Tx-bv, threshold2);
%         
%         r = ATy + mu1*P1(u+bu) + mu2*P2(v+bv);
%                 
%         if (isAconvfilter)
%             if definedW
%                 x = alphainv*( r - WT( real(ifft2(filter_FFT.*fft2( W(r) ))) ) );
%                 if outer == 1
%                     fprintf('Deconvolution, with frames\n')
%                 end
%             else
%                 x = real( ifft2( filter_FFT.*fft2(r) ) );
%                 if outer == 1
%                     fprintf('Deconvolution, without frames\n')
%                 end
%             end
%             numAt = numAt + 1;
%         else
%             if isAmatrix
%                 x = alphainv*(r - invAAT*r);
%                 if outer == 1
%                     fprintf('A is a matrix, with frames\n')
%                 end
%                 numA = numA + 1;
%                 numAt = numAt + 1;
%             else
%                 if isAmask
%                     
%                     if isAfftmask
%                         x = alphainv*(r - (1/(alpha+1))*A(r));
%                         fprintf('A is a spatial domain mask\n')
%                         numA = numA + 1;
%                         numAt = numAt + 1;
%                         
%                     else
%                         x = alphainv*(r - invAAT*A(r));
%                         fprintf('A is a spatial domain mask\n')
%                         numA = numA + 1;
%                         %numAt = numAt + 1;
%                     end
%                 else
%                     if (isinvAAT)
%                         x = alphainv*( r -  AT(invAAT(A(r))) );
%                         numA = numA + 1;
%                         numAt = numAt + 1;
%                     else
%                     end
%                 end
%             end
%                         
%         end
%         
%     end
%     
%     P1Tx = P1T(x);
%     P2Tx = P2T(x);
%     
%     bu = bu + (u - P1Tx);
%     bv = bv + (v - P2Tx);
%     
%     resid =  y-A(x);
%     numA = numA + 1;
%     objective(outer+1) = 0.5*(resid(:)'*resid(:)) + tau1*phi1(u) + tau2*phi2(v);
%     
%     if compute_mse
%         err = x-true;
%         mses(outer+1) =  (err(:)'*err(:))/numel(x);
%     end
%     
%     %distance(outer) = norm(x(:)-z(:),2)/sqrt(norm(x(:),2)^2 + norm(z(:),2)^2);
%     
%     if (outer>1)
%         % take no more than maxiter iterations
%         switch stopCriterion
%             case 1
%                 % compute the stopping criterion based on the relative
%                 % variation of the objective function.
%                 criterion(outer) = abs(objective(outer+1)-objective(outer))/objective(outer);
%             case 2
%                 % compute the stopping criterion based on the relative
%                 % variation of the estimate.
%                 criterion(outer) = abs(norm(x(:)-xprev(:))/norm(x(:)));
%             case 3
%                 % continue if not yet reached target value tolA
%                 criterion(outer) = objective(outer+1);
%             otherwise
%                 error(['Unknown stopping criterion']);
%         end
%         
%          if ( criterion(outer) < tolA )
%              if verbose
%                 fprintf('\niter = %d, obj = %3.3g, stop criterion = %3.3g, ( target = %3.3g )\t', outer, objective(outer+1), criterion(outer), tolA)
%                 if compute_mse
%                     fprintf('MSE = %g',mses(outer+1))
%                 end
% 
%                 fprintf('\nConvergence reached.\n')
%              end
%              times(outer+1) = cputime - t0;
%              break;
%          end
%          
%     end
%     
%     if verbose
%         fprintf('\niter = %d, obj = %3.3g, stop criterion = %3.3g, ( target = %3.3g )\t', outer, objective(outer+1), criterion(outer), tolA)
%         if compute_mse
%             fprintf('MSE = %g',mses(outer+1))
%         end
%     end
%     
%     times(outer+1) = cputime - t0;
% end
% 
% %times(outer) = cputime - t0;
% 
% % if (outputvar == 2)
% %     x_estimate = z;
% % else
% %     x_estimate = x;
% % end