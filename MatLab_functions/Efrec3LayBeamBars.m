function [maxef,Mrt,c]=Efrec3LayBeamBars(Mu,fc,E,h,b,distrRebar,...
                                     listRebarDiam,vSep,b_rec,h_rec,beta1)

%------------------------------------------------------------------------
% Syntax:
% [maxef,Mrt,c]=Efrec3LayBeamBars(Mu,fc,E,h,b,nb3,ab3,...
%                                 b_rec,h_rec,beta1)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: Calculates the structural efficiency of beam cross-section 
% according to the applied load conditions.
% 
% OUTPUT: maxEf,Mr,c:      is the structural efficiency of the reinforced 
%                          beam cross-section (0-1), the resistant beding
%                          moment and the neutral axis depth, respectively
%
% INPUT:  load_conditions: vector as [nload,Mu] size: nloads x 2
%
%         factor_fc:       is determined by de applicable design code. 
%                          The ACI 318-19 specifies it as 0.85
%
%         beta1:           is determined as preescribed by the ACI 318 code
%                          (see documentation)
%
%         h_rec,b_rec:     is the concrete cover along the height dimension
%                          and the width cross-section dimension,
%                          respectively
%
%         h,b:             cross-section dimensions
%
%         ast,asc:         are the reinforcement steel area in tension and
%                          compression, respectively
%
%         E:               is the Elasticity Modulus of reinforcing steel
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2023-02-05
% Copyright (c)  Faculty of Engineering
%                Autonomous University of Queretaro, Mexico
%------------------------------------------------------------------------

fdpc=fc*0.45;

d=h-h_rec;
mumax=max(abs(Mu));

cUno=0.001; 
cDos=h;
fr=0;
[Root]=bisectionMr3LayerBeamsRebar(cUno,cDos,fr,E,distrRebar,...
    listRebarDiam,h,b,b_rec,h_rec,vSep,fdpc,beta1,0.001);  

c=Root(1);
Mrt=Root(3);

factor_resistance=0.87;
Mrt=Mrt*factor_resistance;

maxef=mumax/Mrt;