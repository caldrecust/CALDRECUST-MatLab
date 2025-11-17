function [Mr,Fr]=strElasConcISRecBlock(x,fcu,h,b,Cc,fy,Es,tdistr,tlist)

%------------------------------------------------------------------------
% Syntax:
% [Mr,Fr]=strElasConcBarsRecBlock(x,fcu,h,b,hrec,fy,Es,distrRebar,listRebarDiam)
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
% OUTPUT: sumFrMr:      array that contains the output [Frs, Mrs;
%                                                        Frc, Mrc]
%                       corresponding to the sum of resistant forces.
%                       row 1: sum of resistant actions of steel.
%                       row 2: sum of resistant actions of concrete.
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
dvs=10;
bp=b-2*Cc-2*dvs;
d=h-Cc;

if fcu<=45 % Clause 6.1.2.4.b (HK-2013)
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
nb=length(tlist);
fsc=0; fst=0;
asc=0; ast=0;
azc=0; azt=0;
for i=1:nb
    abi=tlist(i)*bp;
    yi=tdistr(i,1);
    di=0.5*h-yi; %di
    
    %% Reinforcing steel strain model (elastic)
    e=ecu/x*(di-x); %epsilum (steel strain - linear distribution)
    ss=strRebarElas1(e,fy,Es); % linear stress-strain curve for steel
    fs=0.87*ss*abi;
    if fs<0 % steel in compression
        fsc=fsc+fs; % sum of forces of steel in compression
        asc=asc+abi; % sum of rebar area in compression
        azc=azc+abi.*di; 
    else
        fst=fst+fs; % sum of forces of steel in tension
        ast=ast+abi; % sum of rebar area in tension
        azt=azt+abi.*di;
    end
end
dst=azt/ast; % centroid location of reinforcement in tension (distance
             % from the upper most layer of concrete)
if asc>0
    dsc=azc/asc; % centroid location of rebar in compression (distance from the 
                 % upper most layer of concrete)
    
    Mrs=fsc*(dst-dsc); % contribution of steel to bending resistance

else
    Mrs=0;
end

Mrc=frc.*(dst-s/2); % contribution of concrete to bending resistance

Mr=abs(Mrc)+abs(Mrs); % Total bending resistance

Fcomp=frc+fsc; % compression forces
Ften=fst; % tension forces

Fr=Fcomp+Ften; % Total axial load forces

% --------------------------------- End -----------------------------