function [elemConc]=casoConcretoParabEN1992(c,fcm,b,h)

%------------------------------------------------------------------------
% Syntax:
% elemConc=casoConcretoParabEC1992(c,fcm,b,h)
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: To compute the contribution of resistance of the concrete
% compression zone of a rectangular beam cross-section, regarding axial 
% and bending forces. A parabolic strain-stress curve for the concrete,
% according to the EN1992.
% 
% OUTPUT: elemConc: vector that contains the output [Fc, Mc] of 
%                   resistant axial and bending forces
%
% INPUT:  c:        is the reduced depth of neutral axis of the
%                   cross-section in question
%
%         fcm:      is the concrete's compressive strength 
%
%         b,h:      are the cross-section dimensions
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2025-02-05
% Copyright (c)  School of Engineering
%                The HKUST
%------------------------------------------------------------------------

Ecm=22*(fcm/10)^0.3;
if c>h
    c=h;
end
ec1=0.7*fcm^0.31;
if ec1>2.8
    ec1=2.8;
end
ec1=ec1/1000;
k=1.05*Ecm*abs(ec1*1000)/fcm;

ecu1=0.0035;
ec=ecu1;
np=60;
dc=c/np;

eta=ec/ec1;
fc=fcm*(k*eta-eta^2)/(1+(k-2)*eta)*dc/2;
sumfcy=fc*(0.5*dc);

for i=1:np
    ca=dc*i;
    eta=ec/ec1;
    
    if i<np
        fci=fcm*(k*eta-eta^2)/(1+(k-2)*eta)*dc;
    elseif i==np
        fci=fcm*(k*eta-eta^2)/(1+(k-2)*eta)*dc/2;
    end
    fc=fc+fci;
    ec=ecu1-ecu1/c*ca;
    sumfcy=sumfcy+fci*(ca-0.5*dc);
end

yc=sumfcy/fc;

frc=-b*fc;
mrc=-frc*(0.5*h-yc);

elemConc=[frc mrc];
