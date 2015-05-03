function [C, ss, M, X,Ye] = ppca_mv(Ye,d,dia)
%
% implements probabilistic PCA for data with missing values, 
% using a factorizing distribution over hidden states and hidden observations.
%
%  - The entries in Ye that equal NaN are assumed to be missing. - 
%
% [C, ss, M, X, Ye ] = ppca_mv(Y,d,dia)
%
% Y   (N by D)  N data vectors
% d   (scalar)  dimension of latent space
% dia (binary)  if 1: printf objective each step
%
% ss  (scalar)  isotropic variance outside subspace
% C   (D by d)  C*C' +I*ss is covariance model, C has scaled principal directions as cols.
% M   (D by 1)  data mean
% X   (N by d)  expected states
% Ye  (N by D)  expected complete observations (interesting if some data is missing)
%
% J.J. Verbeek, 2006. http://lear.inrialpes.fr/~verbeek
%

[N D]       = size(Ye); % N observations in D dimensions
threshold   = 1e-4;     % minimal relative change in objective funciton to continue    
M    = mean(Ye);                 
Ye = Ye - repmat(M,N,1);

% =======     Initialization    ======
rand("seed",0)
C     = rand(D,d);
traceC=trace(C)
CtC   = C'*C;
tracectc=trace(CtC)
tracectc=trace(inv(CtC))
traceY=trace(Ye)
ss = rand(1,1);
ss

count = 1; 
old   = Inf;
while count          %  ============ EM iterations  ==========      
   
    Minv = inv( ss* eye(d) + CtC );    % ====== E-step, (co)variances   =====
	 traceSx = trace(Minv)
	 X = Ye * C * Minv';
	 fprintf('CDEBUG Y=%f C=%f Minv=%f \n', trace(Ye) , trace(C) , trace(Minv));
	 traceX = trace(X)
    XtX = X'*X + ss * Minv;                              % ======= M-step =====
	 tracesumxtx=trace(XtX)
	 %fprintf('CDEBUG Y=%f Yp=%f X=%f div=%f \n', trace(Ye) , trace(Ye') , trace(X), trace(inv(XtX)));
    C      = (Ye'*X)  / (XtX);    
	 traceC=trace(C)
    CtC    = C'*C;
	 traceCtC=trace(CtC)
    ss     = ( sum(sum( (Ye).^2 )) + trace(XtX*CtC) ) /(N*D); 
	 xcty = 0;
	 for i = 1:N
	   xcty += X(i,:) * C' * Ye(i,:)';
	 endfor
	 ss -= 2 * xcty / (N*D);
	 ss
    
    %objective = N*D + N*(D*log(ss) +trace(Minv)-log(det(Minv)) ) +trace(XtX);
	 objective = ss;
           
    rel_ch    = abs( 1 - objective / old );
    old       = objective;
    
    count = count + 1;
    %if ( rel_ch < threshold) && (count > 5); count = 0;end
    if ( rel_ch < threshold) && (count > 5); count = 0;end
    if dia; fprintf('Objective:  %.2f    relative change: %.5f \n',objective, rel_ch ); end
    
end             %  ============ EM iterations  ==========



%C = orth(C);
%X = Ye*C; 
%Ye = X * C';

%C = orth(C);
%[vecs,vals] = eig(cov(Ye*C));
%[vals,ord] = sort(diag(vals),'descend');
%vecs = vecs(:,ord);
%C = C*vecs;
%X = Ye*C;
 
%[c, ~, ~] = svd(C);
%X = Ye * c(:,1:d);
%[u, ~ ,~] = svd(X');
%X = X*u;



% add data mean to expected complete data
Ye = Ye + repmat(M,N,1);
