% Efficiency_Beam_RebarDesign_Ex01
%----------------------------------------------------------------
% PURPOSE 
%    To determine the structural efficiency of a
%    reinforced rectangular beam cross-section.
%
%----------------------------------------------------------------
%
% LAST MODIFIED: L.F.Veduzco    2023-07-03
% Copyright (c)  School of Engineering
%                The Hong Kong University of Science and Technology (HKUST)
%----------------------------------------------------------------

clc
clear all

%% Geometry
b=280; % cross-section width
h=510; % cross-section height

hrec=50; % concrete cover on vertical direction
brec=30; 
d=h-hrec; % effective cross-section height
bp=b-2*brec;
hp=h-2*hrec;

nb=12;
nRebarsSup=3;
nRebarsInf=4;
nRebarsLeft=2;
nRebarsRight=3;

db1=12;
db2=16;
db3=10;
db4=20;
dvs=10;

barDiamList(1:nRebarsSup,1)=db1;
barDiamList(nRebarsSup+1:nRebarsSup+nRebarsInf,1)=db2;
barDiamList(nRebarsSup+nRebarsInf+1:nRebarsSup+nRebarsInf+nRebarsLeft,1)=db3;
barDiamList(nRebarsSup+nRebarsInf+nRebarsLeft+1:nb,1)=db4;

[distrRebar,sephor1,sephor2,sepver1,sepver2]=distrRebarAsymmCols(b,h,...
[brec,hrec],nb,nRebarsSup,nRebarsInf,nRebarsLeft,nRebarsRight,db1,db2,...
db3,db4,dvs);

plotRebarColRec(distrRebar,h,b,barDiamList)