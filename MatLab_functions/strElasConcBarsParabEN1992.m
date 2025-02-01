function eleMec=strElasConcBarsParabEN1992(c,fdpc,h,b,E,barIndecesComp,...
                            barIndecesTen,disposition_rebar,rebarAvailable)

%------------------------------------------------------------------------
% Syntax:
% eleMec=strElasConcBarsParabEN1992(c,fdpc,h,b,E,barIndecesComp,...
%                     barIndecesTen,disposition_rebar,rebarAvailable)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: To compute the sum of resistant forces of a beam cross-section, 
% considering the contribution of rebars over the cross-section and concrete
% in compression.
% 
% OUTPUT: eleMec:           vector that contains the output [Fs, Ms;
%                                                            Fc, Mc]
%
% INPUT:  barIndecesTen,    Vectors that contain the type of rebar for the
%         barIndecesComp:   optimal option both in tension and compression,
%                           respectively. The vectors size is of one column 
%                           with nrebar rows containing a number between 1 
%                           and 7 according to the available commercial 
%                           rebar types stated by default
%
%         c:                neutral axis depth
%
%         a:                reduced neutral axis depth as a=beta1*c
%
%         E:                Modulus of Elasticity of reinforcing steel
%
%         disposition_rebar:local coordinates of rebars laid out over the 
%                           beam cross-section
%
%         rebarAvailable:   data base of the available commercial types of 
%                           rebar
%
%         fdpc=0.45f'c:     according to ACI 318
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2023-02-05
%                School of Engineering
%                The Hong Kong University of Science and Technology (HKUST)
%------------------------------------------------------------------------
 
arreglo_total=[barIndecesComp;barIndecesTen];

nv=length(arreglo_total);

eMecRebar(:,2)=disposition_rebar(:,1);
eMecRebar(:,3)=disposition_rebar(:,2);

% [dA, xposition, yposition, d, eps, eE, Fr, Mr]

sumaM=0;
sumaF=0;
for i=1:nv
    eMecRebar(i,1)=rebarAvailable(arreglo_total(i),1).^2.*pi./4;
    eMecRebar(i,4)=0.5*h-eMecRebar(i,3); %compression
    eMecRebar(i,5)=0.0035/c*(eMecRebar(i,4)-c); %epsilum
    
    if(eMecRebar(i,5)>0.00217)
        eMecRebar(i,5)=0.00217; %compression
    end
    eMecRebar(i,6)=eMecRebar(i,5)*E;
    eMecRebar(i,7)=eMecRebar(i,6)*eMecRebar(i,1);
    eMecRebar(i,8)=eMecRebar(i,7)*(eMecRebar(i,4)-0.5*h);
    
    sumaF=sumaF+eMecRebar(i,7);
    sumaM=sumaM+eMecRebar(i,8);
end

elemAc=[sumaF sumaM];
elemConc=casoConcretoParabEN1992(c,fdpc,b,h);

eleMec=[elemAc;
        elemConc];
      