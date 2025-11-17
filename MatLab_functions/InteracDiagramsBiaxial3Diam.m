function [diagrama,eccxy,mexef,eficiencia,cxy,Mr]=InteracDiagramsBiaxial3Diam...
    (b,h,Cc,Es,npts,fcu,barDiamList,distrRebar,loadConditions)
                                    
%------------------------------------------------------------------------
% Syntax:
% [diagrama,mexef,eficiencia,cxy]=diagramasDisposicion(As,b,h,E,npuntos,...
%        fdpc,nv,beta,ov,av,disposicion_varillado,load_conditions)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: To compute the interaction diagram of a symmetric rebar option, 
% as well as the structural efficiency given certain load conditions.
% 
% OUTPUT: diagrama:             is the interaction diagram data
%
%         maxef:                is the critical structural efficiency 
%                               corresponding to the critical load condition
%
%         eficiencia:           is a table containing the structural 
%                               efficiency analysis data: size = [nload,8],
%                               in format: 
%           _____________________________________________________
%           [nload,Pu,Mux,Muy,P{Rx},M{Rx},P{Ry},M{Ry},efficiency]
%           _____________________________________________________
%
%         cxy:                  are the neutral axis depth values 
%                               corresponding to the critical load condition, 
%                               for both axis directions: [cx,cy]
%
% INPUT:  b,h:                  given cross-section dimensions
%
%         av,ov,nv:             are the rebar area and number of rebar of 
%                               the current rebar option, and the number of
%                               rebars, respectively
%
%         load_conditions:      is the array containing the load conditions:
%                               size = [nload,4] in format: [nload,Pu,Mux,Muy]
%
%         beta:                 is determined as specified in code ACI,
%                               according to the f'c used
%
%         fdpc:                 equal to 0.85f'c according to code
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2023-02-05
% Copyright (c)  School of Engineering
%                HKUST
%------------------------------------------------------------------------

if npts<3
    disp('Error: the number of points for the Interaction Diagram must be 3 or higher');
    return;
end
    
As=sum(barDiamList.^2*pi/4);

diagrama=zeros(npts,3);
cPtDiag=zeros(npts,2);

fy=Es*0.00217;
Ac=b*h;
pot=0.87*As*fy;

poc=(0.45*fcu*(Ac)+0.87*fy*As);
df=(pot+poc)/(npts-1);

diagrama(1,1)=-poc;
diagrama(npts,1)=pot;

diagrama(1,2)=0;
diagrama(npts,2)=0;

diagrama(1,3)=0;
diagrama(npts,3)=0;

eccxy(1,1)=0;
eccxy(1,2)=0;

cPtDiag(1,1)=4*h;
cPtDiag(1,2)=4*h;

rebar=[distrRebar(:,1) distrRebar(:,2)];
dimensionsCol=[b h];
for ax=1:2
    if (ax==2)
        b=dimensionsCol(2);
        h=dimensionsCol(1);
        distrRebar(:,1)=-rebar(:,2);
        distrRebar(:,2)=rebar(:,1);
    end
    
    for i=1:npts-1
        diagrama(i+1,1)=(-poc+i*df);
        fr=diagrama(i+1,1);
        c1=0.001;
        c2=4*h;
 
        [raiz]=rootMrColsRebar(c1,c2,fr,Es,fy,distrRebar,...
                                barDiamList,h,b,fcu,0.001);
        diagrama(i+1,1+ax)=raiz(3);

        cPtDiag(i+1,ax)=raiz(1);
        
        %%%%%%%%%%%%%%%%%%%%%%%% Eccentricities %%%%%%%%%%%%%%%%%%%%
        eccxy(i+1,ax)=diagrama(i+1,1+ax)/diagrama(i+1,1);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%% Eficiencies %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loadConditions(:,3:4)=abs(loadConditions(:,3:4)); % by only consisering
                                                    % positive moments is
                                                    % enouth for symmetric
                                                    % designs -----------
[mexef,eficiencia,cxy,Mr]=effRecColsLSBi2UniHK(diagrama,loadConditions,...
    fcu,b,h,Cc,cPtDiag);
