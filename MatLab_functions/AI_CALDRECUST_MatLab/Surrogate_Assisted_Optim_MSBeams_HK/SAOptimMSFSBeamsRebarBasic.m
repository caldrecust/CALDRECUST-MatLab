function [volRebar,volRebarSpans,LenRebarL,LenRebarM,LenRebarR,sepRebarSpans,EffSpans,...
  MrSpans,cSpans,ListRebarDiamLeft,ListRebarDiamMid,ListRebarDiamRight,...
  DistrRebarLeft,DistrRebarMid,DistrRebarRight,nbLMRspan,totalnbSpan,CFAspans,...
  nbcut3secSpan,nbLowLeftSpans,nbTopMidSpans,nblowRightSpans,CSrebar,ccSpans]=...
  SAOptimMSFSBeamsRebarBasic(b,h,span,brec,hrec,hagg,pmin,pmax,sepMin,...
  fcu,load_conditions,fy,cutxLoc,Ao3,Wfac,dbc)

nspans=size(load_conditions,1);
dvs=10;

sepRebarSpans=[];
ListRebarDiamLeft=[];
ListRebarDiamMid=[];
ListRebarDiamRight=[];
DistrRebarLeft=[];
DistrRebarMid=[];
DistrRebarRight=[];
nbcut3secSpan=[];

LenRebarL=[];
LenRebarM=[];
LenRebarR=[];

% Consequetive spans
feasibleVec=ones(1,nspans);
ccSpans=0;
nbcc=zeros(1,3);
nblow=zeros(1,3);
abc=dbc.^2*pi/4;
for i=1:nspans
    i1=(i-1)*3+1;
    i2=i*3;
    Aos3=Ao3(i1:i2,1);

    abc=dbc.^2*pi/4;
    
    sepMin1l=sepMin(1);
    sepMin1m=sepMin(2);
    
    nbmaxl=fix((b(i)-2*brec-2*dvs+sepMin1l)/(dbc(1)+sepMin1l));
    nbmaxm=fix((b(i)-2*brec-2*dvs+sepMin1m)/(dbc(2)+sepMin1m));

    [nb1l,nb2l,nb3l,isfeasible]=nbSection3layer(nbmaxl,Aos3(1),abc(1));
    [nb1m,nb2m,nb3m,isfeasible]=nbSection3layer(nbmaxm,Aos3(2),abc(2));
    
    nblm=[nb1l,nb2l,nb3l,nb1m,nb2m,nb3m];

    [bestPerformance,bestLenRebar,bestsepRebar,NbCombo9,bestEfL,bestEffMid,...
    bestEffRight,bestMrLeft,bestMrMid,bestMrRight,cbestLeft,cBestMid,cBestRight,...
    bestListRebarDiamLeft,bestListRebarDiamMid,bestListRebarDiamRight,...
    bestDistrRebarLeft,bestRebarDistrMid,bestDistrRebarRight,bestnbcut3sec,...
    nblowRight,nbTopMid,nblowLeft,CFA,cc]=SAMSBeamsRebarLM1DSection(b(i),h(i),...
    span(i),brec,hrec,hagg,pmin,pmax,sepMin,fcu(i),load_conditions(i,:),fy,...
    cutxLoc(i,:),dbc,Aos3,Wfac,nblm);
    
    if i<nspans
        if Aos3(3,1)>Ao3(i2+1,1)
            % Bending of rebars at the right-end cross-section
            [vSep,vmaxSepm]=sepMinMaxHK13(dbc(1),hagg,1);
            [nbnew1,nbnew2,nbnew3,Abcut]=BendCutRebarSecBeamAs1DiamSection...
                (NbCombo9(7:9),dbc(1),Ao3(i2+1,1),b(i),h(i),hrec,vSep);
    
            nbcc=[nbnew1,nbnew2,nbnew3];
        else
            nbcc=NbCombo9(7:9);
        end
    else
        nbcc=NbCombo9(7:9);
    end
    nblow=nblowRight;
    
    nbcut3secSpan=[nbcut3secSpan;
                    bestnbcut3sec];

    volRebarSpans(i)=bestPerformance;

    sepRebarSpans=[sepRebarSpans;
                   bestsepRebar];
    
    EffSpans(i,:)=[bestEfL,bestEffMid,bestEffRight];

    MrSpans(i,:)=[bestMrLeft,bestMrMid,bestMrRight];
    cSpans(i,:)=[cbestLeft,cBestMid,cBestRight];

    nbleft=length(bestListRebarDiamLeft);
    ListRebarDiamLeft=[ListRebarDiamLeft;
                      bestListRebarDiamLeft];

    nbMid=length(bestListRebarDiamMid);
    ListRebarDiamMid=[ListRebarDiamMid;
                      bestListRebarDiamMid];

    nbright=length(bestListRebarDiamRight);
    ListRebarDiamRight=[ListRebarDiamRight;
                        bestListRebarDiamRight];

    DistrRebarLeft=[DistrRebarLeft;
                    bestDistrRebarLeft];

    DistrRebarMid=[DistrRebarMid;
                   bestRebarDistrMid];

    DistrRebarRight=[DistrRebarRight;
                   bestDistrRebarRight];
    
    LenRebarL=[LenRebarL;
                bestLenRebar(1:sum(NbCombo9(1:3)))];
    LenRebarM=[LenRebarM;
                bestLenRebar(sum(NbCombo9(1:3))+1:sum(NbCombo9(1:6)))];
    LenRebarR=[LenRebarR;
                bestLenRebar(sum(NbCombo9(1:6))+1:sum(NbCombo9))];

    nbLMRspan(i,:)=NbCombo9;

    totalnbSpan(i,:)=[nbleft,nbMid,nbright];
    nbLowLeftSpans(i,:)=nblowLeft;
    nbTopMidSpans(i,:)=nbTopMid;
    nblowRightSpans(i,:)=nblowRight;
    ccSpans=ccSpans+cc;
    
    if CFA==0
        feasibleVec(1,i)=0;
    end
    CFAspans(i)=CFA;
end

volRebar=sum(volRebarSpans);
if sum(feasibleVec)==nspans % There must be a solution for all spans
                            % for the whole optimization design to be
                            % feasible
    CSrebar=mean(CFAspans);
else
    CSrebar=0;
end
end