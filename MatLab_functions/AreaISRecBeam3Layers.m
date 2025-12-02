function As=AreaISRecBeam3Layers(t,b,Cc)

%------------------------------------------------------------------------
% Syntax:
% As=AreaISRecBeam3Layers(t,b,Cc)
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: mm,N,MPA.
%
%------------------------------------------------------------------------
% PURPOSE: To compute the cross-section area of an idealized smeared
% reinforcement t of one or more layers.
% 
% OUTPUT: As:       idealized reinforcement cross-section area (mm^2)
%
% INPUT:  t:        vector containing the width values of the idealized 
%                   smeared reinforcement (m)
%
%         h:        is the cross-section width dimension
%       
%         Cc:       concrete cover (mm)
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2025-02-05
%                School of Engineering
%                The Hong Kong University of Science and Technology (HKUST)
%------------------------------------------------------------------------

dvs=10;
bp=b-2*Cc-2*dvs;

As=sum(t.*bp);
end