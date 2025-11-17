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
b=500; % cross-section width
h=550; % cross-section height

hrec=50; % concrete cover on vertical direction
brec=30; 
d=h-hrec; % effective cross-section height

fcu=30;
dvs=10;
bp=b-2*brec-2*dvs;
hp=h-2*hrec-2*dvs;

nb1=2;
nb2=4;
nb3=2;

db1=12;
db2=16;
db3=10;

comboDbRebar=[db1,db2,db3];
comboNbRebar=[2*nb1,2*nb2,2*nb3];
nb=sum(comboNbRebar);

[distrRebar,barDiamList,sephor1,sepver1]=distrRebarSym3Cols(b,h,[brec,hrec],...
    nb,nb2,nb3,db1,db2,db3,dvs);

%% Loads
loadConditions=[1 -9.95e4 1.26e8 0.97e8]; % [nload, Pu, Mx, My]

%% Structural efficiency
Es=200e3;

[diagram,eccxy,maxef,eff,cxy,Mr]=InteracDiagramsBiaxial3Diam(b,h,hrec,Es,...
    100,fcu,barDiamList,distrRebar,loadConditions);

plotInteracDiagSecRebarCols(loadConditions,diagram,distrRebar,...
                        h,b,barDiamList);
                    