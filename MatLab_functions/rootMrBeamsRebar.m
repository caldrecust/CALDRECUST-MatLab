function [raiz]=rootMrBeamsRebar(c1,c2,fr,Es,fy,distrRebar,...
                                listRebarDiam,h,b,hrec,fcu,ea)

%------------------------------------------------------------------------
% Syntax:
% [raiz]=rootMrBeamsRebar(c1,c2,fr,Es,fy,distrRebar,listRebarDiamh,h,b,hrec,...
%                           fcu,ea)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: To determine the neutral axis depth and resistant bending moment 
% of reinforced beam cross-section taking on account both the steel in 
% compression and steel in tension. The bisection method is used as the
% root method for the pre-established equilibrium condition sum F=0.
% 
% OUTPUT: raiz:         vector that contains the output [c,sum Fi,MR]
%
% INPUT:  c1,c2:        as a closed root method, the bisection method need
%                       two initial root values to start the iterations.
%                       One value should be smaller than the expected real
%                       root and the other should be larger.
%
%         hrec:         are the concrete cover parameters horizontally and
%                       vertically, respectively 
%
%         fcu:          is the concrete's compressive strength
%
%         ea:           is the approximation root error
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2025-02-05
% Copyright (c)  School of Engineering
%                HKUST
%------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%------------- f(l)------------%%%%%%%%%%%%%%%%%%%%%

[Mr,Fr]=strElasConcBarsRecBlock(c1,fcu,h,b,hrec,fy,Es,distrRebar,listRebarDiam);

frt=Fr;
root1=fr-frt;

%%%%%%%%%%%%%%%%%%%%%% f(u) %%%%%%%%%%%%%%%%%%%%%%% 
[Mr,Fr]=strElasConcBarsRecBlock(c2,fcu,h,b,hrec,fy,Es,distrRebar,listRebarDiam);

frt=Fr;
root2=fr-frt;

%%%%%%%%%%%%%%%%%%%%%% f(xr) %%%%%%%%%%%%%%%%%%%%%%
c=c2-(root2*(c1-c2)/(root1-root2));

if c<=0
    c=0.000001;
end

[Mr,Fr]=strElasConcBarsRecBlock(c,fcu,h,b,hrec,fy,Es,distrRebar,listRebarDiam);

frt=Fr;
rootc=fr-frt;

%%%%%%%%%%%%%%%%%%%%%%%%%%% begin loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%
it1=0;
it2=0;

cu=c2;
es=abs((c-cu)/c);
while(es>ea)

    if((root1*rootc)<0)
        c2=c;
        
        [Mr,Fr]=strElasConcBarsRecBlock(c2,fcu,h,b,hrec,fy,Es,distrRebar,listRebarDiam);

        frt=Fr;
        root2=fr-frt;

        it1=it1+1;

    elseif((root1*rootc)>0)
        c1=c;
        
        [Mr,Fr]=strElasConcBarsRecBlock(c1,fcu,h,b,hrec,fy,Es,distrRebar,listRebarDiam);
        
        frt=Fr;
        root1=fr-frt;

        it2=it2+1;
    end

    cu=c;

    c=c2-(root2*(c1-c2)/(root1-root2));
    if c<=0
        c=0.000001;
    end
    
    [Mr,Fr]=strElasConcBarsRecBlock(c,fcu,h,b,hrec,fy,Es,distrRebar,listRebarDiam);
    
    frt=Fr;
    rootc=fr-frt;

    es=abs((c-cu)/c);
    
end
mrt=Mr;
raiz=[c,frt,mrt];   