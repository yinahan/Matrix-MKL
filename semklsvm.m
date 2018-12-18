function [Sigma,Alpsup,w0,pos,history,obj,status] = semklsvm(K,yapp,C,option,verbose)

% USAGE [Sigma,Alpsup,w0,pos,history,obj,status] = mklsvm(K,yapp,C,option,verbose)
%
% Inputs
%
% K         : NxNxD matrix containing all the Gram Matrix
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
%     Sigma=rand(n,nbkernel);
%     Sigma=Sigma./repmat(sum(Sigma,2),1,nbkernel);
%     Sigma=ones(n,nbkernel);
%     Sigma=Sigma./repmat(sum(Sigma,2),1,nbkernel);
else
    Sigma=option.sigmainit ;
    ind=find(Sigma==0);
end;


if isfield(option,'alphainit');
    alphainit=option.alphainit;
else
    alphainit=[];
end;




%--------------------------------------------------------------------------------
% Options used in subroutines
%--------------------------------------------------------------------------------
% if ~isfield(option,'goldensearch_deltmax');
%     option.goldensearch_deltmax=1e-1;
% end
% if ~isfield(option,'goldensearchmax');
%     optiongoldensearchmax=1e-8;
% end;
% if ~isfield(option,'firstbasevariable');
%     option.firstbasevariable='first';
% end;

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
switch option.algo
    case 'svmclass'
        [xsup,Alpsup,w0,pos,aux,aux,obj] = svmclass([],yapp,C,lambdareg,kernel,kerneloption,verbosesvm,span,alphainit);
        [ff]=ffsvmclass(K,pos,Alpsup,Sigma);
        
%         normf = repmat(sum(f.^2,2),1,size(f,2));
%         f=f./(sqrt(normf)+eps);
    case 'svmreg'
        % for svmreg Alpsup is the vector of [alpha alpha*]
        if ~isfield(option,'svmreg_epsilon')
            error(' Epsilon tube is not defined ... see option.svmreg_epsilon ...');
        end;
        [xsup,ysup,Alpsupaux,w0,pos,Alpsup,obj] = svmreg([],yapp,C,option.svmreg_epsilon,kernel,kerneloption,lambdareg,verbosesvm);
        grad = gradsvmreg(K,Alpsup,yapp,option) ;
end;

% Sigmaold  = Sigma ;
Alpsupold = Alpsup ;
w0old     = w0;
posold    = pos ;
ffold = ff;

telapsed = toc(tstart);
history.telapsed = [history.telapsed telapsed];
history.sigma= [history.sigma;Sigma];
history.obj=[history.obj obj];

objold = obj;
obj = 0;
%------------------------------------------------------------------------------%
% Update Main loop
%------------------------------------------------------------------------------%

while loop & nloopmax >0 %& obj<objold
    
%     if option.p ==1
%         [SigmaNew]=gammaupdate(K,pos,Alpsup,f,Sigma);
%     elseif option.p >1
%         [SigmaNew]=gammaupdate_p(pos,f,Sigma,option.p);
%     end

    SigmaNew = (ff.^(1/(1+option.p)))./repmat(sum(ff.^(option.p/(1+option.p)))^(1/option.p),size(Sigma));

    
    alphainit=zeros(size(yapp));
    alphainit(pos)=yapp(pos).*Alpsup;
    kerneloption.matrix=sumKbeta(K,SigmaNew);
    [xsup,Alpsup,w0,pos,aux,aux,obj] =...
        svmclass([],yapp,C,lambdareg,kernel,kerneloption,verbosesvm,span,alphainit);
    [ff]=ffsvmclass(K,pos,Alpsup,SigmaNew);
    telapsed = toc(tstart);
    
    if obj<=objold
        nloop = nloop+1;
        objold = obj;
        normek=-gradsvmclass(K,pos,Alpsup,C,yapp,option); % 0.5*Alpsup'*K(pos,pos,i)*Alpsup;
        dualitygap=(obj +   max(normek) - sum(abs(Alpsup)))/obj;
        Sigma = SigmaNew;
        history.dualitygap=[history.dualitygap dualitygap];
        history.obj=[history.obj obj];
        history.sigma= [history.sigma;Sigma];
        history.telapsed = [history.telapsed telapsed];
        Alpsupold = Alpsup;
        w0old = w0;
        posold = pos;
        ffold = ff;
    else
        obj = objold;
        w0 = w0old;
        Alpsup = Alpsupold;
        pos = posold;
        ff = ffold;
%         save([option.path,'/newhistory',num2str(C),'.mat'], 'history'); 
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
        loop = 0;
        status=2;
        fprintf(1,'maximum number of iterations reached\n')
    end;
    if nloop < option.miniter & loop==0
        loop=1;
    end;
end

% save([option.path,'/seMKL/history',num2str(C),'.mat'], 'history');  