function [diagramIntAxis1,pot,poc,neuAx,newdistrRebar,...
    newCoordCorners,gamma]=InteracDiagAxColSym3Diam...
    (npdiag,db3l,b,h,fy,fcu,Es,nb1,nb2,nb3,distrRebar,Mux,Muy)
%------------------------------------------------------------------------
% Syntax:
% [diagramIntAxis1,pot,poc,cvectorX,newdispositionRebar,...
%  newCoordCorners,newCP,gamma]=InteractionDiagramAxis...
%  (npdiag,comborebar,b,h,fy,fdpc,beta1,E,number_rebars_sup,...
%  number_rebars_inf,number_rebars_left,number_rebars_right,...
%  rebarAvailable,dispositionRebar,Mux,Muy)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%------------------------------------------------------------------------
% PURPOSE: To compute the interaction diagram with respect to a rotated
%          axis of a rectangular cross-section asymmetrically reinforced. 
% 
% OUTPUT: diagramIntAxis1:      is the array containing the interaction 
%                               diagram data whose rebars in tension are at
%                               the bottom of the cross-section. See
%                               Documentation
%
%         pot,poc:              are the max axial load resistance in
%                               tension and compression, respectively
%
%         newCP:                are the Plastic Center depth over the
%                               cross-section (with respect to the upper 
%                               outer most concrete fibre) in both axis
%                               directions. See documentation
% 
%         cvectorX:             is the neutral axis depth values along the
%                               interaction diagram of the axis in quest
%                               (from the upper cross-section corner
%                               downwards)
%
%         newdispositionRebar:  is the array containing the local
%                               coordinates of the rebars distributed over
%                               the rotated rectangular cross-section
%
%         gamma:                is the angle of rotation for the
%                               cross-section
%
% INPUT:  b,h:                  cross-section initial dimensions
%
%         rebarAvailable:       rebar database consisting of an array of 
%                               size [7,2] by default in format: 
%                               [#rebar,diam]
%
%         number_rebars_sup,
%         number_rebars_inf,
%         number_rebars_left,
%         number_rebars_right:  number of rebars to placed on each of the 
%                               cross-section boundaries
%
%         comborebar:           vector of size [1,4] which are the
%                               types of rebar for each of the 
%                               four cross-section boundaries (upper 
%                               boundary, lower boundary, left side and 
%                               right side, respectively)
%
%         fdpc:                 is the reduced concrete's compressive
%                               strength by the application of the
%                               reduction resistance factor of 0.85
%                               (according to the ACI 318 code)
%
%         npdiag:               number of points to be computed for the
%                               definition of the interaction diagram
%
%         dispositionRebar:     is the vector containing the rebar local 
%                               coordinates over the cross-section [x,y] 
%
%         Mux,Muy:              Bending moment components with respect to
%                               each cross-section local axis
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2023-02-05
% Copyright (c)  Faculty of Engineering
%                Autonomous University of Queretaro, Mexico
%------------------------------------------------------------------------
if npdiag<3
    disp('Error: the number of points for any axis of the Interaction') 
    disp('Surface must be at least 3');
    return;
end

db1=db3l(1);
db2=db3l(2);
db3=db3l(3);

ab1=db1.^2*pi/4;
ab2=db2.^2*pi/4;
ab3=db3.^2*pi/4;

diagramIntAxis1=zeros(npdiag,3);
neuAx=zeros(npdiag,1);

As=4*ab1+2*nb2*ab2+2*nb3*ab3;

%% Rotation of the reinforced cross-section

alpha=rad2deg(atan(Mux/Muy));
if Muy<=0 && Mux>=0 || Muy<=0 && Mux<=0
    gamma=(90-alpha)+180;
elseif Muy>=0 && Mux<=0 || Muy>=0 && Mux>=0
    gamma=90-alpha; % This is the angle for the section to be rotated at
                    % so that the resultant moment of Mx and My is now Mx'
end

[newdistrRebar,newCoordCorners]=rotReCol(Mux,Muy,distrRebar,b,h);
                                
%% Interaction diagram - Direction X'

poc=(0.87*As*fy+0.45*fcu*(b*h));
pot=0.87*As*fy;
df=(poc+pot)/(npdiag-1);

diagramIntAxis1(1,1)=-poc;
diagramIntAxis1(1,2)=1e-7;

neuAx(1,1)=4*max(newCoordCorners(:,2));
neuAx(npdiag,1)=0;

ea=0.001;
for i=1:npdiag-1
    diagramIntAxis1(i+1,1)=-poc+i*df;
    c1=0.001; 
    c2=4*max(newCoordCorners(:,2));
    fr=diagramIntAxis1(i+1,1);

    [root]=rootMrColSym3DiamRot(c1,c2,fr,Es,h,b,fcu,...
        ea,nb1,nb2,nb3,db1,db2,db3,newdistrRebar,newCoordCorners,gamma);
    
    diagramIntAxis1(i+1,2)=root(3);
    neuAx(i+1,1)=root(1);

    %%%%%%%%%%%%%%%%%%%%%%%%% Eccentricities %%%%%%%%%%%%%%%%%%%%%%%%%%

    diagramIntAxis1(i+1,3)=diagramIntAxis1(i+1,2)/...
                           diagramIntAxis1(i+1,1);
end
