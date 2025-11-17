function [Eff,Mrt,c]=EfRecBeamt(Mu,fcu,Es,fy,h,b,tdistr,tlist,Cc)

%------------------------------------------------------------------------
% Syntax:
% [Eff,Mrt,c]=EfRecBeamt(Mu,fcu,Es,fy,h,b,distrRebar,listRebarDiam,hrec)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: Calculates the structural efficiency of beam cross-section 
% according to the applied load conditions.
% 
% OUTPUT: Eff,Mr,c:        is the structural efficiency of the reinforced 
%                          beam cross-section (0-1), the resistant beding
%                          moment and the neutral axis depth, respectively
%
% INPUT:  Mu:              bending load for design
%
%         fcu:             is the concrete's compressive strength
%
%         hrec:            is the concrete cover along the height dimension
%                          and the width cross-section dimension,
%                          respectively
%
%         h,b:             cross-section dimensions
%
%         Es:              is the Elasticity Modulus of reinforcing steel
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2023-02-05
% Copyright (c)  School of Engineering
%                HKUST
%------------------------------------------------------------------------

Mumax=max(abs(Mu));

cUno=0.001; 
cDos=h;
fr=0;
[Root]=rootMrBeamst(cUno,cDos,fr,Es,fy,tdistr,tlist,h,b,Cc,fcu,0.001);  

c=Root(1);
Mrt=Root(3);

Eff=Mumax/Mrt;
