function [Mr,Fr]=strElasBars1RecBlockColRot(x,fcu,h,b,fy,Es,distrRebar,...
                      listRebarDiam,RotCornerSec,depthRotCP,gamma)

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

coordCP=max(RotCornerSec(:,2))-depthRotCP;

%% Concrete model
if fcu<=60
    ecu=0.0035; % concrete's ultimate strain (Clause 6.1.2.4)
else
    ecu=0.0035-6e-5*sqrt(fcu-60);
end
s=0.9*x;
hrot=2*depthRotCP;
if s>hrot
    s=hrot;
end

[frc,yc]=casoConcretoRecBlockRot(s,fcu,b,h,RotCornerSec,gamma);

%% Resistance of the overall section

nb=length(listRebarDiam);
fst=0;
Mrs=0;
for i=1:nb
    abi=pi/4*listRebarDiam(i,1)^2;
    yi=distrRebar(i,2);
    di=depthRotCP-yi; %di
    
    %% Reinforcing steel strain model (elastic)
    
    e=ecu/x*(di-x); %epsilum (steel strain - linear distribution)
    ss=strRebarElas1(e,fy,Es); % linear stress-strain curve for steel
    fs=0.87*ss*abi;
    fst=fst+fs; % sum of forces of steel
    %Mrs=Mrs+fs*(coordCP-yi); % contribution of steel to bending resistance
    Mrs=Mrs+fs*(di-depthRotCP); % contribution of steel to bending resistance
    
end
%Mrc=-frc*(depthRotCP-yc); % contribution of concrete to bending resistance
Mrc=frc*(yc-depthRotCP); % contribution of concrete to bending resistance

Mr=Mrc+Mrs; % Total bending resistance
Fr=frc+fst; % Total axial load forces
% --------------------------------- End -----------------------------