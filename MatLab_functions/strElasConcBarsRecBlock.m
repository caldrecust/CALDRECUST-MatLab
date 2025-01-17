function [eleMec]=strElasConcBarsRecBlock(x,fcu,h,b,hrec,...
                                           Es,distrRebar,listRebarDiam)

%------------------------------------------------------------------------
% Syntax:
% [eleMec]=strElasConcBarsRecBlock(x,fcu,h,b,hrec,Es,distrRebar,listRebarDiam)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: N,mm.
%
%------------------------------------------------------------------------
% PURPOSE: To compute the sum of resistant forces of a beam cross-section,
% considering the contribution of steel in tension, steel in compression 
% and concrete in compression. The simiplified rectangular stress block at
% ULS is considered.
% 
% OUTPUT: eleMec:       vector that contains the output [Fs, Ms;
%                                                        Fc, Mc]
%
% INPUT:  distrRebar:   is the vector size: nbars x 2, containig the local
%                       coordinates of the rebars over the cross-section
%
%        listRebarDiam: is the list of rebar diameter sizes to be placed
%                       over the concrete element cross-section
%
%         b,h:          are the cross-section dimensions
%
%         fcu:          Is the concrete's compressive strength value
%
%         x:            is the neutral axis depth (mm)
%
%         Es:           is the Young's modulus of stel
%
%         hrec:         is the vertical concrete cover
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2025-02-05
% Copyright (c)  School of Engineering
%                HKUST
%------------------------------------------------------------------------
d=h-hrec;
fcu=fcu/0.45;
if fc<=45 % Clause 6.1.2.4.b (HK-2013)
    if x>0.5*d
        x=0.5*d;
    end
    s=0.9*x;
elseif 45<fcu && fcu<=70
    if x>0.4*d
        x=0.4*d;
    end
    s=0.8*x;
elseif 70<fcu && fcu<=100
    if x>0.33*d
        x=0.33*d;
    end
    s=0.72*x;
end

%% Concrete model
if fcu<=60
    ecu=0.0035; % concrete's ultimate strain (Clause 6.1.2.4)
else
    ecu=0.0035-6e-5*sqrt(fcu-60);
end
frc=casoConcretoRecBlock(s,fcu,b);

%% Resistance of the overall section
nb=length(listRebarDiam);
fsc=0;
fst=0;
asc=0;
ast=0;
azc=0;
azt=0;
for i=1:nb
    abi=pi/4*listRebarDiam(i,1)^2;
    yi=distrRebar(i,2);
    di=0.5*h-yi; %di
    
    %% Reinforcing steel strain model (elastic)
    e=ecu/x*(di-x); %epsilum (steel strain - linear distribution)
    s=strRebarElas(e,fy,Es); % linear stress-strain curve for steel
    fs=0.87*s*abi;
    if fs<0 % steel in compression
        fsc=fsc+fs; % sum of forces of steel in compression
        asc=asc+abi; % sum of rebar area in compression
        azc=azc+abi*di; 
    else
        fst=fst+fs; % sum of forces of steel in tension
        ast=ast+abi; % sum of rebar area in tension
        azt=azt+abi*di;
    end
end
dst=azt/ast; % centroid location of reinforcement in tension (distance
             % from the upper most layer of concrete)
dsc=azc/asc; % centroid location of rebar in compression (distance from the 
             % upper most layer of concrete)

Mrs=fsc*(dst-dsc); % contribution of steel to bending resistance
Mrc=frc*(dst-a/2); % contribution of concrete to bending resistance
Mr=Mrc+Mrs; % Total bending resistance

Fcomp=frc+fsc; % compression forces
Ften=fst; % tension forces

Fr=Fcomp+Ften; % Total axial load forces

eleMec=[fsc+fsc Mrs; % Steel resistance (axial, bending)
        frc     Mrc]; % Concrete resistance (axial, bending)

% --------------------------------- End -----------------------------