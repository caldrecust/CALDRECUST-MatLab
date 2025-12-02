function [volRebar,volRebarSpans,LenRebarL,LenRebarM,LenRebarR,sepRebarSpans,EffSpans,...
  MrSpans,cSpans,ListRebarDiamLeft,ListRebarDiamMid,ListRebarDiamRight,...
  DistrRebarLeft,DistrRebarMid,DistrRebarRight,nbLMRspan,totalnbSpan,CFAspans,...
  CSAssemCutSpan,nbcut3secSpan,nbLowLeftSpans,nbTopMidSpans,nblowRightSpans,...
  CSrebar,ccSpans]=SAMOOptimMSFSBeamsRebarSimple(b,h,span,brec,hrec,hagg,pmin,...
  pmax,sepMin,fcu,load_conditions,fy,cutxLoc,Ao3,Wfac,dbc)

nspans=size(load_conditions,1);
dvs=10;
Es=fy/0.00217;

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
    
    Aos3(1,1)=Ao3(i1,1);
    Aos3(2,1)=Ao3(i1+1,1);
    if nspans>1 
        if i< nspans
            Aos3(3,1)=max(Ao3(i2+1,1)+Ao3(i2,1));
        else
            Aos3(3,1)=Ao3(i2,1); 
        end
    else
        Aos3(3,1)=Ao3(i2,1);
    end
    rr1=b(i)-2*brec-2*dvs+sepMin(1);
    rr2=b(i)-2*brec-2*dvs+sepMin(2);
    rr3=b(i)-2*brec-2*dvs+sepMin(3);

    rr21=dbc(1)+sepMin(1);
    rr22=dbc(2)+sepMin(2);
    rr23=dbc(3)+sepMin(3);

    nbmax1l=max([fix((rr1)/(rr21)),2]);
    nbmax2l=max([fix((rr2)/(rr22))]);
    nbmax3l=max([fix((rr3)/(rr23))]);
    nbmaxl=[nbmax1l,nbmax2l,nbmax3l];

    [nb1l,nb2l,nb3l,isfeasible]=nb3DSec1(nbmaxl,Aos3(1,1),abc(1:3),nbcc);

    rr1=b(i)-2*brec-2*dvs+sepMin(4);
    rr2=b(i)-2*brec-2*dvs+sepMin(5);
    rr3=b(i)-2*brec-2*dvs+sepMin(6);

    rr21=dbc(4)+sepMin(4);
    rr22=dbc(5)+sepMin(5);
    rr23=dbc(6)+sepMin(6);

    nbmax1r=max([fix((rr1)/(rr21)),2]);
    nbmax2r=max([fix((rr2)/(rr22))]);
    nbmax3r=max([fix((rr3)/(rr23))]);
    nbmaxm=[nbmax1r,nbmax2r,nbmax3r];

    [nb1m,nb2m,nb3m,isfeasible]=nb3DSec1(nbmaxm,Aos3(2,1),abc(4:6),nblow);

    nblm=[nb1l,nb2l,nb3l,nb1m,nb2m,nb3m];
    
    [bestPerformance,bestLenRebar,bestsepRebar,NbCombo9,bestEfL,bestEffMid,...
    bestEffRight,bestMrLeft,bestMrMid,bestMrRight,cbestLeft,cBestMid,cBestRight,bestListRebarDiamLeft,...
    bestListRebarDiamMid,bestListRebarDiamRight,bestDistrRebarLeft,bestRebarDistrMid,...
    bestDistrRebarRight,bestnbcut3sec,nblowLeft,dblowLeft,nbTopMid,...
    dbTopMid,nblowRight,dblowRight,CFA,cc]=SACutRedistrOptimRecBeam3DSec...
    (load_conditions(i,:),fcu(i),Es,h(i),b(i),span(i),dbc,hagg,brec,hrec,pmin,pmax,sepMin,cutxLoc(i,:),...
    Aos3,Wfac,nblm);
    
    nbcc=NbCombo9(7:9);
    nblow=nblowRight;
    
    nbcut3secSpan=[nbcut3secSpan;
                    bestnbcut3sec];
    db9Spans(i,:)=[dbc,dbc(1:3)];
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
    
    if CFA(1)==0
        feasibleVec(1,i)=0;
    end
    CSAssemCutSpan(i,:)=CFA(2:3);
    CFAspans(i)=CFA(1);
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