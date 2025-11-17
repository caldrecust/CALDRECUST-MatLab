function [bestPerformance,bestEffR,bestMrR,bestcRight,bestnb,bestsepRebar,...
    bestListDiam,bestRebarDistr,bestdbc]=NbRebarR1DSec...
    (Mur,fc,h,b,hagg,brec,hrec,pmin,pmax,sepRebarleft,distrRebarComp,...
    listRebarDiamComp,nbAfterCut3,nb3l,dbc,Aos3)

%------------------------------------------------------------------------
% Syntax:
% [c_best,bestMr,bestEf,best_area,tbest,h]=PSO3layerBeamsRebar(b,h,duct,...
%    b_rec,h_rec,vSep,fc,Mu,fy,graphConvergencePlot)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: SI - (Kg,cm)
%                  US - (lb,in)
%------------------------------------------------------------------------
% PURPOSE: To determine an optimal reinforcement area for a given beam 
% cross-section with specified initially dimensions (b,h) through the SGD 
% method.
% 
% OUTPUT: c_best,bestMr,bestEf: The neutral axis depth for the optimal 
%                               design, the resistant bending moment for 
%                               the optimal design,
%
%         best_area,tbest:      The optimal reinforcement area, the optimal 
%                               t width of the ISR
%
%         h:                    The final cross-section height in case it 
%                               is modified from the given initial proposal 
%                               value
%
% INPUT:  load_conditions:      vector as [nload,Mu] size: nloads x 2
%
%         factor_fc:            is determined by de applicable design code. 
%                               The ACI 318-19 specifies it as 0.85
%
%         duct:                 is the ductility demand parameter, with 
%                               possible values of 1,2 or 3, for low 
%                               ductility, medium ductility or high 
%                               ductility respectively
%
%         h_rec,b_rec:          is the concrete cover along the height dimension
%                               and the width cross-section dimension, respectively
%                               (cm)
%
%         h,b:                  cross-section dimensions (cm)
%
%         E:                    is the Elasticity Modulus of reinforcing steel
%                               (Kg/mm2)
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2023-07-03
% Copyright (c)  Faculty of Engineering
%                Autonomous University of Queretaro, Mexico
%------------------------------------------------------------------------

Es=200e3;
dvs=10;
bp=b-2*brec-2*dvs;

%% Number of rebars
bestPerformance=1e10;
bestEffR=0;

[sepMin,sepMax1m]=sepMinMaxHK13(dbc,hagg,0);
%nbmaxm=fix((b-2*brec-2*dvs-2*sepMin+sepMin)/(dbc+sepMin));

rr1=b-2*brec-2*dvs-sepMin;
rr2=dbc+sepMin;

nbmaxr=fix((rr1)/(rr2));

ab=dbc^2*pi/4;

nbro=ceil(Aos3(3)/ab); % minimum required number of bars for left cross-section

if nbro<2
    nbro=2;
end

isfeasibleRightDistr=false;
if nbro>3*nbmaxr
    isfeasibleRightDistr=false;
    nb1r=0;
    nb2r=0;
    nb3r=0;
else
    isfeasibleRightDistr=true;
    if sum(nbAfterCut3)>=nbro % no more rebars are needed
        nb1r=nbAfterCut3(1);
        nb2r=nbAfterCut3(2);
        nb3r=nbAfterCut3(3);
        
    else
        if nbro<=nbmaxr 
            if nbAfterCut3(1)==2
                nb1r=nbro;
                if nb1r<nbAfterCut3(1)
                    nb1r=2;
                end
                nb2r=0;
                nb3r=0;
                
            elseif nbAfterCut3(1)>2
                nb1r=nb3l(1);
                
                if nbro-nb1r>0
                    nb2r=nbro-nb1r;
                else
                    nb2r=0;
                end
                
                if nb2r<nbAfterCut3(2)
                    nb2r=nbAfterCut3(2);
                end
                
                nb3r=0;
                
                if nb3r<nbAfterCut3(3)
                    nb3r=nbAfterCut3(3);
                end
            end
            if mod(nb1r,2)==0 && mod(nb2r,2)~=0
                nb2r=nb2r+1;
            end
            
            if mod(nb2r,2)==0 && mod(nb3r,2)~=0
                nb3r=nb3r-1;
            end
            
        elseif nbro>nbmaxr
            nb1r=nb3l(1);
            if nbro-nb1r>=nbmaxr
                nb2r=nb3l(1);
                nb3r=nbro-nb1r-nb2r;
                if nb3r<nbAfterCut3(3)
                    nb3r=nbAfterCut3(3);
                end
                if mod(nb2r,2)==0 && mod(nb3r,2)~=0
                    nb3r=nb3r+1;
                end
                
            elseif nbro-nb1r<nbmaxr
                nb2r=nbro-nb1r;
                if nb2r<nbAfterCut3(2)
                    nb2r=nbAfterCut3(2);
                end
                nb3r=0;
                if mod(nb1r,2)==0 && mod(nb2r,2)~=0
                    nb2r=nb2r+1;
                end
                
            end
        end
    end
end

nbr=[nb1r,nb2r,nb3r];

%% Rebar separation

sepRebar1=(bp)/(nbr(1)-1);
if nbr(2)>1
    sepRebar2=(bp)/(nbr(2)-1);
else
    sepRebar2=1e5;
end
if nbr(3)>1
    sepRebar3=(bp)/(nbr(3)-1);
else
    sepRebar3=1e5;
end
sepRebarRight=[sepRebar1,sepRebar2,sepRebar3];

for ii=1:3
    if nb3l(ii)-nbAfterCut3(ii)==0 && nbAfterCut3(ii)>2 % if there werent cuts
        sepRebarRight(ii)=sepRebarleft(ii);
    end
end

[Abr,EffR,MrR,cRight,xBest,ListDiam,RebarDistr,isfeasibleConstr]=...
DistrEffConstrR1DSec(Mur,fc,Es,h,b,hagg,brec,hrec,pmin,pmax,...
nbAfterCut3,distrRebarComp,listRebarDiamComp,nb3l,dbc,nbr);

if (Abr<bestPerformance && isfeasibleRightDistr && isfeasibleConstr)

    bestdbc=dbc;
    bestPerformance=Abr;
    bestEffR=EffR;
    bestMrR=MrR;
    bestcRight=cRight;
    bestnb=xBest;
    bestsepRebar=sepRebarRight;
    bestListDiam=ListDiam;
    bestRebarDistr=RebarDistr;
end
        
if bestEffR==0
    bestPerformance=1e10;
    bestEffR=0;
    bestMrR=0;
    bestcRight=0;
    bestnb=zeros(1,3);
    bestsepRebar=zeros(1,3);
    bestListDiam=0;
    bestRebarDistr=[];
    bestdbc=zeros(1);
end