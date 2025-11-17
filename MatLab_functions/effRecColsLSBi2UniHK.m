function [maxef,tabEff,cxy,Mr]=effRecColsLSBi2UniHK(diagrama,loadConditions,fcu,b,...
                                        h,Cc,cPtsDiag)
%------------------------------------------------------------------------
% Syntax:
% [maxef,tabEff,cxy]=effRecColsLinearSearch(diagrama,...
%                             load_conditions,pot,poc,c_vector_bar)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: To compute the structural efficiency of a symmetrically 
%          reinforced column cross-section, given multiple load conditions.
%          The Bresler's formula is used (for biaxial bending compression)
%          and the Euclidean distance is used (for uniaxial bending
%          compression).
% 
% OUTPUT: maxef:                is the critical structural efficiency 
%                               corresponding to the critical load condition
%
%         tabEff:               is a table containing the structural 
%                               efficiency analysis data: size = [nload,8],
%                               in format: 
%           _____________________________________________________
%           [Pu,Mux,Muy,P{Rx},M{Rx},P{Ry},M{Ry},efficiency]
%           _____________________________________________________
%
%         cxy:                  are the neutral axis depth values 
%                               corresponding to the critical load condition, 
%                               for both axis directions: [cx,cy]
%
% INPUT:  diagrama:             is the interaction diagram data
%
%         load_conditions:      is the array containing the load conditions:
%                               size = [nload,4] in format [nload,Pu,Mux,Muy]
%
%         pot,poc:              are the max resistant axial force in tension
%                               of reinforcement steel (concrete is not 
%                               considered) and compression of the whole 
%                               reinforced cross-section area (both concrete
%                               area and rebar area are considered) 
%
%         c_vector_bar:         is the array containing the neutral axis 
%                               depth values for both cross-section axis 
%                               directions for all interaction diagram 
%                               points: size = (npoints+2) x 2
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2023-02-05
% Copyright (c)  School of Engineering
%                HKUST
%------------------------------------------------------------------------

%% RESISTANCE COMPUTATION 
hp=h-Cc;
bp=b-Cc;
cax=[];
maxef=-inf;
imax=1;
nconditions=size(loadConditions,1);
tabEff=zeros(nconditions,6);
for j=1:nconditions
    pu=loadConditions(j,2);
    mux=loadConditions(j,3);
    muy=loadConditions(j,4);
    fbeta=fbetaNCols(pu,fcu,b,h);
    if mux/hp>=muy/bp
        mux=mux+fbeta*hp/bp*muy;
        ax=1;
        mu=mux;
    elseif mux/hp<muy/bp
        muy=muy+fbeta*bp/hp*mux;
        ax=2;
        mu=muy;
    end

    penu=1/(mu/pu); % NOTE: The slope is the one taken for comparison

    tabEff(j,1)=pu;
    tabEff(j,2)=mux;
    tabEff(j,3)=muy;

    k=1;

    ydos=diagrama(k,1);
    ytres=diagrama(k+1,1);
    xdos=diagrama(k,1+ax);
    xtres=diagrama(k+1,1+ax);
    pendos=ydos/xdos;
    pentres=ytres/xtres;
    while pendos<penu && pentres<penu
        k=k+1;
        ydos=diagrama(k,1);
        ytres=diagrama(k+1,1);
        xdos=diagrama(k,1+ax);
        xtres=diagrama(k+1,1+ax);
        pendos=ydos/xdos;
        pentres=ytres/xtres;
        if pendos<penu && pentres>penu
            break;
        end
    end

    mr=(((ydos-ytres)/(xtres-xdos))*xtres+ytres)/...
        (pu/mu-((ytres-ydos)/(xtres-xdos)));
    if mr>max(diagrama(:,1+ax))
        [mr,k]=max(diagrama(:,1+ax));
    end
    cdos=cPtsDiag(k,ax);
    ctres=cPtsDiag(k+1,ax);

    c=(cdos+ctres)*0.5;
    
    cax=[cax;c];
        
    if mr==0
        mr=0.00001;
    end
    pr=pu/mu*mr;

    tabEff(j,4)=pr;
    tabEff(j,5)=mr;

    ef=sqrt(pu^2+mu^2)/sqrt(pr^2+mr^2);
    if (ef>=maxef)
        imax=j;
        maxef=ef;
    end
    tabEff(j,6)=ef;
end

% To store the neutral axis depth values corresponding to the critical
% load condition
cxy=cax(imax);
Mr=tabEff(imax);