function [frc,mrc]=casoConcretoTBlock(s,fcu,be,hf,bw,dst)

%------------------------------------------------------------------------
% Syntax:
% frc = casoConcretoRecBlock(s,fcu,b,h)
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: To compute the contribution of resistance of the concrete
% compression zone of a rectangular beam cross-section, regarding axial 
% and bending forces.
% 
% OUTPUT: frc:       resistant axial force of concrete in compression
%
% INPUT:  s:        is the reduced depth of neutral axis of the
%                   cross-section in question
%
%         fcu:      value of f'c (not factorized) 
%
%         b:        is the cross-section width dimension
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2025-02-05
%                School of Engineering
%                The Hong Kong University of Science and Technology (HKUST)
%------------------------------------------------------------------------

%% Area in compression

if s<=hf
    ac=s*be;
    ca=0.5*s;
else
    ac=hf*be+(s-hf)*bw;
    
    % Centroid of the concrete zone in compression
    ca=((hf*be)*(0.5*hf)+(s-hf)*bw*(hf+0.5*(s-hf)))/(be*hf+(s-hf)*bw);
end

%% Resistance
fdpc=0.45*fcu;
frc=-ac*fdpc;

mrc=frc*(dst-ca); % contribution of concrete to bending resistance
