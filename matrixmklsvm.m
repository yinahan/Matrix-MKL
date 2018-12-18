function [Sigma,Alpsup,w0,pos,history,obj,status] = matrixmklsvm(K,yapp,C,option,verbose)

% USAGE [Sigma,Alpsup,w0,pos,history,obj,status] = mklsvm(K,yapp,C,option,verbose)
%
% Inputs
%
% K         : NxNxD matrix containing all the Gram Matrix
% yapp      : Class label of the training set
% C         : SVM hyperparameter
% option    : mkl algorithm hyperparameter
%
%       option.nbitermax : maximal number of iterations (default 1000)
%       option.algo      : selecting algorithm svmclass (default) or svmreg
%       option.seuil     : threshold for zeroing kernel coefficients
%                          (default 1e-12) during the first
%                          option.seuilitermax
%       option.seuilitermax : threshold weigth when the nb of iteration is
%                           lower than this value
%       option.sigmainit  : initial kernel coefficient (default average)
%       option.alphainit : initial Lagrangian coefficient
%       option.lambdareg : ridge regularization added to kernel's diagonal
%                          for SVM (default 1e-10)
%       option.verbosesvm : verbosity of SVM (default 0) see svmclass or
%                           svmreg
%       option.svmreg_epsilon : epsilon for SVM regression (mandatory for
%                         svm regression)
%       option.numericalprecision    : force to 0 weigth lower than this
%                                     value (default = eps)
%
% Outputs
%
% Sigma         : the weigths
% Alpsup        : the weigthed lagrangian of the support vectors
% w0            : the bias
% pos           : the indices of SV
% history        : history of the weigths
% obj           : objective value
% status        : output status (sucessful or max iter)

[n] = length(yapp);
if ~isempty(K)
    if size(K,3)>1
        nbkernel=size(K,3);
        if option.efficientkernel==1
            K = build_efficientK(K);
        end;
    elseif option.efficientkernel==1 & isstruct(K);
        nbkernel=K.nbkernel;     
    end;
else
    error('No kernels defined ...');
end;

if ~isfield(option,'nbitermax');
    nloopmax=1000;
else
    nloopmax=option.nbitermax;
end;
if ~isfield(option,'algo');
    option.algo='svmclass';
end;
if ~isfield(option,'seuil');
    seuil=1e-12;
else
    seuil=option.seuil;
end
if ~isfield(option,'seuilitermax')
    option.seuilitermax=20;
end;
if ~isfield(option,'seuildiffsigma');
    option.seuildiffsigma=1e-5;
end
if ~isfield(option,'seuildiffconstraint');
    option.seuildiffconstraint=0.05;
end

if ~isfield(option,'lambdareg');
    lambdareg=1e-10;
    option.lambdareg=1e-10;
else
    lambdareg=option.lambdareg;
end


if ~isfield(option,'numericalprecision');
    option.numericalprecision=0;
end;

if ~isfield(option,'verbosesvm');
    verbosesvm=0;
    option.verbosesvm=0;
else
    verbosesvm=option.verbosesvm;
end

if ~isfield(option,'sigmainit') || isempty(option.sigmainit);
    Sigma=ones(1,nbkernel)/nbkernel;
else
    Sigma=option.sigmainit ;
    ind=find(Sigma==0);
end;


if isfield(option,'alphainit');
    alphainit=option.alphainit;
else
    alphainit=[];
end;


%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%
kernel       = 'numerical';
span         = 1;
nloop = 0;
loop = 1;
status=0;
numericalaccuracy=1e-9;
% goldensearch_deltmaxinit= option.goldensearch_deltmax;


tic
tstart = tic;
%-----------------------------------------
% Initializing SVM
%------------------------------------------
SumSigma=sum(Sigma,2);
history.obj=[];
history.sigma=[];
history.KKTconstraint=[];
history.dualitygap=[];
history.telapsed=[];

if ~isempty(K)
    kerneloption.matrix=sumKbeta(K,Sigma);
else
    error('No kernels defined ...');
end;

[xsup,Alpsup,w0,pos,aux,aux,obj] = svmclass([],yapp,C,lambdareg,kernel,kerneloption,verbosesvm,span,alphainit);
[w2sigma] = w2svmclass(K,pos,Alpsup,Sigma,option);
        

Alpsupold = Alpsup ;
w0old     = w0;
posold    = pos ;
w2sigmaold = w2sigma;

telapsed = toc(tstart);
history.telapsed = [history.telapsed telapsed];
history.sigma= [history.sigma;Sigma];
history.obj=[history.obj obj];

objold = obj;
obj = 0;
%------------------------------------------------------------------------------%
% Update Main loop
%------------------------------------------------------------------------------%

while loop & nloopmax >0 
    

%%%%%%%%%%%%%%%%%%%%%%%%%%  Equation (11) %%%%%%%%%%%%%%%%%%%%%%%
%     SigmaNew = (w2sigma)./repmat(sum(w2sigma.^option.p)^(1/(2*option.p))*sum(w2sigma.^option.r)^(1/(2*option.r)),size(Sigma));
 
%%%%%%%%%%%%%%%%%%%%%%%%%%  Equation (12)  %%%%%%%%%%%%%%%%%%%%%%%
    SigmaNew = (w2sigma)./repmat(sum(w2sigma.^option.p)^(1/(option.p+option.r))*sum(w2sigma.^option.r)^(1/(option.p+option.r)),size(Sigma));



    alphainit=zeros(size(yapp));
    alphainit(pos)=yapp(pos).*Alpsup;
    kerneloption.matrix=sumKbeta(K,SigmaNew);
    [xsup,Alpsup,w0,pos,aux,aux,obj] =...
        svmclass([],yapp,C,lambdareg,kernel,kerneloption,verbosesvm,span,alphainit);
    [w2sigma] = w2svmclass(K,pos,Alpsup,SigmaNew,option);
    
    telapsed = toc(tstart);
    
    if obj<=objold
        nloop = nloop+1
        objold = obj;
        normek=-gradsvmclass(K,pos,Alpsup,C,yapp,option); 
        dualitygap=(obj +   max(normek) - sum(abs(Alpsup)))/obj;
        Sigma = SigmaNew;
        history.dualitygap=[history.dualitygap dualitygap];
        history.obj=[history.obj obj];
        history.sigma= [history.sigma;Sigma];
        history.telapsed = [history.telapsed telapsed];
        
        Alpsupold = Alpsup;
        w0old = w0;
        posold = pos;
        w2sigmaold = w2sigma;
    else
        obj = objold;
        w0 = w0old;
        Alpsup = Alpsupold;
        pos = posold;
        w2sigma = w2sigmaold;
        return
    end
    
    %----------------------------------------------------
    % check duality gap
    %----------------------------------------------------
    if  option.stopdualitygap== 1 & dualitygap < option.seuildualitygap
        loop = 0;
        fprintf(1,'Duality gap criteria reached \n');
    end;
    
    %-----------------------------------------------------
    % check nbiteration conditions
    %----------------------------------------------------
    if nloop>=nloopmax ,
        nloop = 0;
        status=2;
        fprintf(1,'maximum number of iterations reached\n')
    end;
    if nloop < option.miniter & loop==0
        loop=1;
    end;
end
