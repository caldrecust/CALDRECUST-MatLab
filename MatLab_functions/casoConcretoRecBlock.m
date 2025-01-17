function frc=casoConcretoRecBlock(s,fcu,b)

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

fdpc=0.45*fcu;

frc=-s*b*fdpc;

