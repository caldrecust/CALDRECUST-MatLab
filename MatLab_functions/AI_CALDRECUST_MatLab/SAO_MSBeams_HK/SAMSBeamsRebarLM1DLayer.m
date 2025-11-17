function [performance,LenRebar,sepRebar,NbCombo9,EffLeft,EffMid,EffRight,MrLeft,MrMid,...
    MrRight,cLeft,cMid,cRight,ListRebarDiamLeft,ListRebarDiamMid,...
    ListRebarDiamRight,DistrRebarLeft,RebarDistrMid,DistrRebarRight,nbcut3sec,...
    nblowRight,nbTopMid,nblowLeft,CFA,const]=SAMSBeamsRebarLM1DLayer(b,h,span,...
    brec,hrec,hagg,pmin,pmax,sepMin,fcu,load_conditions,fy,cutLoc,dbc,Ao3,...
    Wfac,nblm)

%------------------------------------------------------------------------
% Syntax:
% [c_best,bestMr,bestEf,best_area,tbest,h]=PSO3layerBeamsRebar3sec(b,h,duct,...
%    b_rec,h_rec,vSep,fc,Mu,fy,graphConvergencePlot)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: SI - (Kg,cm)
%                  US - (lb,in)
%------------------------------------------------------------------------
% PURPOSE: To determine an optimal reinforcement area for a given beam 
% cross-section with specified initially dimensions (b,h) through the SGD 
% method.
% 
% OUTPUT: c_best,bestMr,bestEf: The neutral axis depth for the optimal 
%                               design, the resistant bending moment for 
%                               the optimal design,
%
%         best_area,tbest:      The optimal reinforcement area, the optimal 
%                               t width of the ISR
%
%         h:                    The final cross-section height in case it 
%                               is modified from the given initial proposal 
%                               value
%
% INPUT:  load_conditions:      vector as [nload,Mu] size: nloads x 2
%
%         factor_fc:            is determined by de applicable design code. 
%                               The ACI 318-19 specifies it as 0.85
%
%         duct:                 is the ductility demand parameter, with 
%                               possible values of 1,2 or 3, for low 
%                               ductility, medium ductility or high 
%                               ductility respectively
%
%         h_rec,b_rec:          is the concrete cover along the height dimension
%                               and the width cross-section dimension, respectively
%                               (cm)
%
%         h,b:                  cross-section dimensions (cm)
%
%         E:                    is the Elasticity Modulus of reinforcing steel
%                               (Kg/cm2)
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2023-07-03
% Copyright (c)  School of Engineering
%                HKUST
%------------------------------------------------------------------------
Es=fy/0.00217;
[performance,LenRebar,sepRebar,NbCombo9,EffLeft,EffMid,...
EffRight,MrLeft,MrMid,MrRight,cLeft,cMid,cRight,ListRebarDiamLeft,...
ListRebarDiamMid,ListRebarDiamRight,DistrRebarLeft,RebarDistrMid,...
DistrRebarRight,nbcut3sec,nblowLeft,dblowLeft,nbTopMid,...
dbTopMid,nblowRight,dblowRight,CFA,const]=SACutRedistrOptimRecBeam3DSec...
(load_conditions,fcu,Es,h,b,span,dbc,hagg,brec,hrec,pmin,pmax,sepMin,cutLoc,...
Ao3,Wfac,nblm);

