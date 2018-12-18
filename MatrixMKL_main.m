
%   Please cite the following papers if you find this work is useful:
%   Han Yina, Yang Yixin, Li Xuelong, Liu Qingyu, Ma Yuanliang, Matrix Regularized Multiple Kernel Learning via (r,p) Norms[J]. 
%   IEEE Transactions on Neural Networks and Learning Systems. Volume: 29, Issue: 10, pp.4997 - 5007. 2018. 
%   Copyright 2018~ , Han Yina. 
%   Author:   Han Yina.  Email£ºyina.han@nwpu.edu.cn
%   Revision: 1.0  
%   Date: 2018/12/13 




clear all
close all

verbose=1;


%------------------------------------------------------
% choosing the stopping criterion
%------------------------------------------------------
options.stopvariation=0; % use variation of weights for stopping criterion
options.stopKKT=0;       % set to 1 if you use KKTcondition for stopping criterion
options.stopdualitygap=0; % set to 1 for using duality gap for stopping criterion

%------------------------------------------------------
% choosing the stopping criterion value
%------------------------------------------------------
options.seuildiffsigma=1e-2;        % stopping criterion for weight variation
options.seuildiffconstraint=0.1;    % stopping criterion for KKT
options.seuildualitygap=0.01;       % stopping criterion for duality gap

%------------------------------------------------------
% Setting some numerical parameters
%------------------------------------------------------
options.goldensearch_deltmax=1e-1; % initial precision of golden section search
options.numericalprecision=1e-8;   % numerical precision weights below this value
% are set to zero
options.lambdareg = 1e-8;          % ridge added to kernel matrix

%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base
% variable in the reduced gradient method
options.nbitermax=500;             % maximal number of iteration
% options.nbitermax=3;
options.seuil=0;                   % forcing to zero weights lower than this
options.seuilitermax=10;           % value, for iterations lower than this one

options.miniter=0;                 % minimal number of iterations
options.verbosesvm=0;              % verbosity of inner svm algorithm
options.efficientkernel=0;         % use efficient storage of kernels

%------------------------------------------------------------------------



classcode=[1 -1];
CC = [0.01 0.1 1 10 100 1000];
LP = [1,2,4,10];
nbiter=20;
ratio=0.8;



load liverdisorder.mat;

data =  liverdisorder;


x = data(:,1:end-1);
y = 2*(data(:,end)==1)-1;

[nbdata,dim]=size(x);

nbtrain=floor(nbdata*ratio);

for lp = 1:length(LP)
    options.p = LP(lp);
    for lr = 1:length(LP)
        options.r = LP(lr);
        for cc = 1:length(CC)
            parameters.C = CC(cc);
            rand('state',0)
            for i=1: nbiter
                i
                [xapp,yapp,xtest,ytest,indice]=CreateDataAppTest(x, y, nbtrain,classcode);
                [xapp,xtest]=normalizemeanstd(xapp,xtest);
                x_norm = [xapp;xtest];
                
                kernelt={'gaussian'};
                variablevec={'single1'};
                for dim = 1:size(xapp,2)
                    kerneloptionvect{dim} = 0.5*mean(pdist(xapp(:,dim), 'euclidean').^2);
                end
                
                [kernel,kerneloptionvec,variableveccell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
                [Weight,InfoKernel]=UnitTraceNormalization(xapp,kernel,kerneloptionvec,variableveccell);
                Ktrain=mklkernel(xapp,InfoKernel,Weight,options);
                
                tic
                
                %%%%%%%%%%% matrix (r,p) MKL %%%%%%%%%%%%%
                
                options.sigmainit = [];
                options.alphainit = [];
                
                %%%%%%% initialization with lp-MKL %%%%%%%
                
                [beta1,w1,b1,posw1,story1,obj1] = semklsvm(Ktrain,yapp,parameters.C,options,verbose);
                options.sigmainit = beta1;
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                
                
                [beta,w,b,posw,story{i},obj] = matrixmklsvm(Ktrain,yapp,parameters.C,options,verbose);
                timelasso(i)=toc;
                
                
                
                Kt = mklkernel(xtest,InfoKernel,Weight,options,xapp(posw,:),beta);
                ypred = Kt*w+b;
                
                
                acc{i}(lp,lr,cc) = mean(sign(ypred)==ytest);
                sv{i}(lp,lr,cc) = length(posw)/length(yapp);
                Obj{i}(lp,lr,cc) = obj;
                
            end
        end
    end
end













