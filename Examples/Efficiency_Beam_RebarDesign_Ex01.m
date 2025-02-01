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

%% Materials
fcu=30; % concrete's compressive strength

fy=500; % Yield stress of steel reinforcement (N/mm2)
Es=fy/0.00217; % Modulus of elasticity of the reinforcing steel

%% Load conditions
loadConditions=[1 270e6]; % N-mm
Mu=loadConditions(1,2);

%% Rebar data
% Database of the commercially available rebar
rebarAvailable=[1 6;
                2 8;
                3 10;
                4 12;
                5 16;
                6 20;
                7 25;
                8 32;
                9 40];
            
% Distribution of rebars over the cross-section
distrRebar=[-bp/2 -hp/2;                   
            -bp/4 -hp/2;
            bp/4 -hp/2;
            bp/2 -hp/2;
            -bp/2 hp/2;
            bp/2 hp/2];
                
RebarDiamList=[32;20;20;32;20;20]; % rebar diameters to use for the reinforcement
                    % in tension (indices from the "rebarAvailable" array)
            
%% Additional design information of interest
ast=sum(RebarDiamList([1:4]',1).^2.*pi./4);
disp('Rebar cross-section area in tension: ');disp(ast);
asc=sum(RebarDiamList([5:6]',1).^2.*pi./4);
disp('Rebar cross-section area in compression: ');disp(asc);

astotal=ast+asc; % Total rebar area
disp('Total rebar cross-section area: ');disp(astotal);

rhos=astotal/(b*h); % Total percentage area
disp('Total percentage of rebar cross-section area: ');disp(rhos);

amin=0.003*b*h; % Min allowed rebar area by code
disp('Min allowed rebar area by code: ');disp(amin);

amax=0.025*b*h; % Max allowed rebar area by code
disp('Max allowed rebar area by code: ');disp(amax);

%% Structural efficiency

[Eff,Mrt,c]=EfRecBeamBars(Mu,fcu,Es,fy,h,b,distrRebar,...
                                     RebarDiamList,hrec);
disp('Bending moment resistance: ');disp(Mrt);
fprintf('Structural efficiency Mu/Mr = %.4f ',Eff); fprintf('\n\n');
disp('Neutral axis depth: ');disp(c);

plotBeamReinforcedSection(h,b,distrRebar,RebarDiamList)
