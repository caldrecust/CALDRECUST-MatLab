function ldb=anchorLenBarTen(fcu,fy,h,Cc,db)

%------------------------------------------------------------------------
% Syntax:
% ldb=anchorLenBarTen(fcu,fy,h,hrec,db)
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: mm,N,MPA.
%
%------------------------------------------------------------------------
% PURPOSE: To compute the achorage length of reinforcement steel bar
% 
% OUTPUT: ldb:      anchorage length (mm)
%
% INPUT:  fy:       yield stress of steel reinforcement MPa
%
%         fcu:      compressive strength of concrete MPa
%
%         h:        is the cross-section height dimension
%       
%         Cc:       concrete cover (mm)
%
%         db:       bar diameter (mm)
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2025-02-05
%                School of Engineering
%                The Hong Kong University of Science and Technology (HKUST)
%------------------------------------------------------------------------

if fcu<=30
    ka=40;
elseif fcu==35
    ka=38;
elseif fcu==40
    ka=35;
elseif fcu==45
    ka=33;
elseif fcu==50
    ka=31;
elseif fcu>=60
    ka=28;
end

d=h-Cc;
ldb1=ka*db+d/2; % Table 8.4 HK 2013

fbu=0.5*sqrt(fcu);
ldb2=0.87*fy/(4*fbu)*db; % Table 8.3 HK 2013

ldb=max([d,ldb1,ldb2]);