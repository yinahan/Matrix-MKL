function [w2eta] = w2svmclass(K,indsup,Alpsup,eta,option)

nsup  = length(indsup);
[n] = size(K,1);

d=size(K,3);
w2eta = zeros(1,d);
for k=1:d;
    %  grad(k) = - 0.5*Alpsup'*Kaux(indsup,indsup)*(Alpsup)  ;
    %         grad(k) = - 0.5*Alpsup'*K(indsup,indsup,k)*(Alpsup) ;
     
    %%%%%% Alpsup 已经有符号了 %%%%%%%%
%     w2(k) = (Alpsup.*label(indsup))'*K(indsup,indsup,k)*(Alpsup.*label(indsup))*(eta(k)^2);
    w2 = Alpsup'*K(indsup,indsup,k)*Alpsup*(eta(k)^2);
    w2eta(k) = w2/((eta(k)/norm(eta,option.p))^option.p+(eta(k)/norm(eta,option.r))^option.r);
end

