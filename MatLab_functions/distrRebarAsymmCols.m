function [dispositionRebar,sephor1,sephor2,sepver1,sepver2]=...
    distrRebarAsymmCols(b,h,rec,nb,nRebarsSup,nRebarsInf,nRebarsLeft,...
    nRebarsRight,db1,db2,db3,db4,dvs)

%------------------------------------------------------------------------
% Syntax:
% [dispositionRebar,sephor1,sephor2,sepver1,sepver2]=...
% dispositionRebarAsymmetric(b,h,rec,nv,nRebarsSup,nRebarsInf,nRebarsLeft,...
%  nRebarsRight,RebarAvailable,dv1,dv2,dv3,dv4)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: To compute the local coordinates of an asymmetric rebar option.
% 
% OUTPUT: dispositionRebar:     are the local coordinates of the optimal 
%                               rebar option
%
%         sephor1,
%         sephor2,
%         sepver1,
%         sepver2:              resultant rebar separation to be compared 
%                               with the minimum one (upper, lower, left
%                               right boundary), respectively
%
% INPUT:  b,h:                  cross-section initial dimensions
%
%         rec:                  are the concrete cover values for each axis
%                               direction of the cross-section
%
%         RebarAvailable:       rebar database consisting of an array of 
%                               size: n# x 2, by default in format: 
%                               [#rebar, diam]
%
%         nRebarsSup,
%         nRebarsInf,
%         nRebarsLeft,
%         nRebarsRight:         number of rebars to placed on each of the 
%                               cross-section boundaries
%
%         op1,op2,op3,op4:      resultant types of rebar for each of the 
%                               four cross-section boundaries (upper 
%                               boundary, lower boundary, left side and 
%                               right side, respectively)
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2023-02-05
% Copyright (c)  Faculty of Engineering
%                HKUST
%------------------------------------------------------------------------

bprima=b-2*rec(1)-2*dvs;
hprima=h-2*rec(2)-2*dvs;

dispositionRebar=zeros(nb,2);

%% Bars - superior part
sephor1=round((bprima-((nRebarsSup)*db1))/(nRebarsSup-1),1);
    
xl1=linspace(-0.5*bprima,0.5*bprima,nRebarsSup);
dispositionRebar(1:nRebarsSup,1)=xl1';
dispositionRebar(1:nRebarsSup,2)=0.5*hprima-0.5*db1;

%% Bars - inferior part
sephor2=round((bprima-((nRebarsInf)*db2))/ (nRebarsInf-1),1);
    
xl2=linspace(-0.5*bprima,0.5*bprima,nRebarsInf);
dispositionRebar(nRebarsSup+1:nRebarsSup+nRebarsInf,1)=xl2';
dispositionRebar(nRebarsSup+1:nRebarsSup+nRebarsInf,2)=-0.5*hprima+0.5*db2;

%% Bars - Left part
sepver1=round((hprima-((nRebarsLeft+1)*db3))/(nRebarsLeft),1);

yl3=linspace(-0.5*hprima+db2+sepver1,0.5*hprima-db1-sepver1,nRebarsLeft);
dispositionRebar(nRebarsSup+nRebarsInf+1:nRebarsSup+nRebarsInf+nRebarsLeft,2)=yl3';
dispositionRebar(nRebarsSup+nRebarsInf+1:nRebarsSup+nRebarsInf+nRebarsLeft,1)=-0.5*bprima+0.5*db3;

%% Bars - Right part
sepver2=round((hprima-((nRebarsRight+1)*db4))/(nRebarsRight),1);

yl4=linspace(-0.5*hprima+db2+sepver2,0.5*hprima-db1-sepver2,nRebarsRight);
dispositionRebar(nRebarsSup+nRebarsInf+nRebarsLeft+1:nb,2)=yl4';
dispositionRebar(nRebarsSup+nRebarsInf+nRebarsLeft+1:nb,1)=0.5*bprima-0.5*db4;
