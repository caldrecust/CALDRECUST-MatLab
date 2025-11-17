function [volRebar,lenRebarL,lenRebarCL]=volRebarDesignBeamCant(nb6l,nb6c,...
    abl6,ab6c,ldL6,nbAfterCut6L,cutLoc,span)

%% Total length of bars (considering cuts)
[lbarL1,lbarL2,lbarL3,lbarL4,lbarL5,lbarL6]=lenRebarRecBeamCuts(nb6l,...
    nbAfterCut6L,span,ldL6+cutLoc(1));

lenRebarL=[lbarL1;lbarL2;lbarL3;lbarL4;lbarL5;lbarL6];
lbarCL1=span;
lbarCL2=span;

lenRebarCL=[lbarCL1;lbarCL2];

%% Rebar volume
volRebar=sum(lbarL1.*abl6(1))+sum(lbarL2.*abl6(2))+sum(lbarL3.*abl6(3))+...
         sum(lbarL4.*abl6(4))+sum(lbarL5.*abl6(5))+sum(lbarL6.*abl6(6))+...
         sum(lbarCL1.*nb6c(1).*ab6c(1))+sum(lbarCL2.*nb6c(2).*ab6c(2));
