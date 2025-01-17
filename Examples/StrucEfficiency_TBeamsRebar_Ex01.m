% StrucEfficiency_TBeamsRebar_Ex01
%----------------------------------------------------------------
% PURPOSE 
%    To analise the structural efficiency of a rebar distribution over a
%    beam element of T cross-section 
%
%----------------------------------------------------------------

% LAST MODIFIED: L.F.Veduzco    2023-07-03
% Copyright (c)  School of Engineering
%                The Hong Kong University of Science and Technology (HKUST)
%----------------------------------------------------------------
clc
clear all

%% Geometry

bw=200; % web width (mm) 

ht=500; % total height (mm)
b=600; % flange width (mm) 
hf=100; % flange height or thickness (mm)
span=500; % mm

cover=50; % concrete cover (mm)
d1=50; 
d2=ht-cover;

%% Materials
fcu=30; % N/mm2
fy=500; % Yield stress of steel reinforcement (N/mm2)

fdpc=fcu*0.45; % reduced f'c
beta1=0.9;

%% Load conditions
load_conditions=[1 249e6]; % N-mm

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
            
% Rebar coordinates
dispositionRebar=[-50 -200;
                   -25 -200;
                   25 -200;
                   50 -200];
               
rebarType=[7;7;7;7]; % index of the rebar diameter to use from the 
                    % database table
            
%% Additional design information of interest
As=sum(rebarAvailable(rebarType,1).^2*pi./4)
rho=As/(bw*d2) % Total percentage area

amin=0.003*bw*d2 % Min allowed rebar area by code
amax=0.025*bw*d2 % Max allowed rebar area by code

%% Analysis of efficiency
[Eff,Mr,cx]=EfRebarTBeams(load_conditions,bw,ht,b,hf,span,fdpc,fy,...
                           rebarType,rebarAvailable,cover,beta1,...
                           dispositionRebar)
                       
TbeamReinforcedSection(bw,ht,b,hf,dispositionRebar,rebarType,...
                                [],rebarAvailable);
