function sigmas=strRebarElas1(e,fy,Es)

%------------------------------------------------------------------------
% Syntax:
% sigmas=strRebarElas1(e,fy,Es)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: To compute the stress of reinforcing steel according to the
% perfect elasto-plastic model
% 
% OUTPUT: sigmas:       stress resistance of reinforcing steel
%
% INPUT:  e:            steel strain
%
%         fy:           yield stress of steel
%
%         Es:           Elastic modulus of steel
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2025-02-05
% Copyright (c)  School of Engineering
%                HKUST
%------------------------------------------------------------------------

esy = fy / Es ;
if abs (e) > esy
    sigmas = fy * e / abs (e) ;
else
    sigmas = e * Es ;
end