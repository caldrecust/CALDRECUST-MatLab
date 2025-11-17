function [Eff,Mrt,c]=EfRebarTBeams(Mu,bw,h,hf,span,fcu,Es,fy,...
                                   listRebarDiam,Cc,distrRebar)
%------------------------------------------------------------------------
% Syntax:
% [Eff,Mr,c]=EfRebarTBeams(load_conditions,bp,ht,ba,ha,Lb,fdpc,...
% rebarType,rebarAvailable,cover,beta1,rebarDisposition)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: SI - (Kg,cm)
%                  US - (lb,in)
%------------------------------------------------------------------------
% PURPOSE: Computes the structural efficiency of a rebar reinforced 
% T-beam cross-section according to the applied load conditions (pure 
% flexure).
% 
% OUTPUT: Eff,Mr,c:        is the structural efficiency of the reinforced 
%                          beam cross-section (0-1), the resistant beding
%                          moment and the neutral axis depth, respectively
%
% INPUT:  load_conditions: vector as [nload,Mu] size: nloads x 2 
%
%         beta1:           is determined as preescribed by the ACI 318 code
%                          (see documentation) according to the f'c value
%
%         fdpc:            is the reduced concrete compressive strength 
%                          (0.85 * f'c ) as prescribed in the ACI 318 code
%
%         ba:              is the effective flange width of the T-beam 
%                          cross-section
%
%         ht:              is total height of the T-beam cross-section
%
%         bp:              is the web width of the T-beam cross-section
%
%         ha:              is the flange thickness of the T-beam
%                          cross-section
%
%         Lb:              is the length of the beam element
%
%         cover:           is the concrete cover for the reinforcement
%
%         rebarAvailable:  is the rebar data base table with each of the
%                          eight-of-an-inch rebars available (from the
%                          smallest to the biggest diameter)
%
%         rebarType:       is the vector containing the rebar diameters'
%                          indices (according to their place in the rebar 
%                          database table)
%
%         rebarDisposition:is the array containing the rebar local
%                          coordinates over the T-beam cross-section
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2023-02-05
% Copyright (c)  Faculty of Engineering
%                Autonomous University of Queretaro, Mexico
%------------------------------------------------------------------------ 

Mumax=max(abs(Mu));

c1=0.001; 
c2=h;
fr=0;
[Root]=rootMrBeamsRebarTBeams(c1,c2,fr,Es,fy,distrRebar,...
                                listRebarDiam,h,hf,bw,span,Cc,fcu,0.001);

c=Root(1);
Mrt=Root(3);

Eff=Mumax/Mrt;