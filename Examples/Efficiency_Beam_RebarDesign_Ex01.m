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
b=300; % cross-section width
h=550; % cross-section height

hrec=50; % concrete cover on each direction

d=h-hrec; % effective cross-section height

%% Materials
fcu=30; % concrete's compressive strength
factor_fc=0.45; % reduction factor for the f'c
fdpc=factor_fc*fcu; % reduced f'c
fy=500; % Yield stress of steel reinforcement (N/mm2)
E=fy/0.0021; % Modulus of elasticity of the reinforcing steel

%% Load conditions
load_conditions=[1 249.9e6]; % KN-mm

%% Rebar data
% Database of the commercially available rebar
rebarAvailable=[6
                8
                10
                12
                16
                20
                25
                32
                40];
            
% Distribution of rebars over the cross-section
dispositionRebar=[-75 -200;                   
                    -25 -200;
                    25 -200
                    75 -200];
                
RebarIndexTen=[7;6;6;7]; % rebar diameters to use for the reinforcement
                    % in tension (indices from the "rebarAvailable" array)
                    
RebarIndexCom=[]; % rebar diameters to use for the reinforcement
                  % in compression (indices from the "rebarAvailable" array)

%% Additional design information of interest
ast=sum(rebarAvailable(RebarIndexTen,1).^2.*pi./4);

astotal=ast % Total rebar area
rho=astotal/(b*d) % Total percentage area

amin=0.003*b*d % Min allowed rebar area by code
amax=0.025*b*d % Max allowed rebar area by code

%% Structural efficiency
[maxef,Mrv,c]=EfcriticalRebarbeams(load_conditions,b,E,fdpc,RebarIndexTen,...
    RebarIndexCom,rebarAvailable,d,hrec,0.9,dispositionRebar)

beamReinforcedSection(h,b,rebarAvailable,dispositionRebar,...
                                RebarIndexCom,RebarIndexTen)