function [ff] = ffsvmclass(K,indsup,Alpsup,eta)

nsup  = length(indsup);
[n] = size(K,1);

d=size(K,3);
ff = zeros(1,d);
for k=1:d;
    %  grad(k) = - 0.5*Alpsup'*Kaux(indsup,indsup)*(Alpsup)  ;
    %         grad(k) = - 0.5*Alpsup'*K(indsup,indsup,k)*(Alpsup) ;
    ff(k) = Alpsup'*K(indsup,indsup,k)*(Alpsup)*(eta(k)^2);
end