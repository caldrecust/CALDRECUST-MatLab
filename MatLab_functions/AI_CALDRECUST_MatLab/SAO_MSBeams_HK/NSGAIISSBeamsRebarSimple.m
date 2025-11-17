function [extrOptPFArea,extrOptPFCS,PF_CFA,PF_VOL,newPop,ff2,genCFA,genVOL,IGDt,IGDv]=...
    NSGAIISSBeamsRebarSimple(b,h,span,brec,hrec,hagg,pmin,pmax,rebarAvailable,...
    fcu,load_conditions,fy,wac,cutLoc,dbcc,nbcc,dblow,nblow,W_PF_REF,...
    CFA_PF_REF,Wfac,Ao3,pccb,MIGDconv,pop_size,gen_max,pltPF)

yp_max=max(W_PF_REF);
np_baseline=length(CFA_PF_REF(:,1));

Es=fy/0.00217;
dvs=10;

%% Variables' range

ndiam=length(rebarAvailable(:,1));

% Identifying existing rebar diameters in left section

indexLayerDiam=[];
for j=1:3
    for i=1:length(rebarAvailable(:,1))
        if dbcc(j)==rebarAvailable(i,2)
            indexLayerDiam=[indexLayerDiam,i];
        end
    end
end

% Identifying layers of rebar valid for optimization for left section

indexLayerOptim=zeros(1,3);
xmaxL=zeros(1,3);
xminL=zeros(1,3);
for i=1:3
    if nbcc(i)==0
        indexLayerOptim(i)=i;
        xmaxL(1,i)=ndiam;
        xminL(1,i)=1;
    else
        indexLayerOptim(i)=i;
        xmaxL(1,i)=indexLayerDiam(i);
        xminL(1,i)=indexLayerDiam(i);
    end
end

% Identifying existing rebar diameters in right section

indexLayerDiam=[];
for j=1:3
    for i=1:length(rebarAvailable(:,1))
        if dblow(j)==rebarAvailable(i,2)
            indexLayerDiam=[indexLayerDiam,i];
        end
    end
end

% Identifying layers of rebar valid for optimization for right section

indexLayerOptim=zeros(1,3);
xmaxR=zeros(1,3);
xminR=zeros(1,3);
for i=1:3
    if nblow(i)==0
        indexLayerOptim(i)=i;
        xmaxR(1,i)=ndiam;
        xminR(1,i)=1;
    else
        indexLayerOptim(i)=i;
        xmaxR(1,i)=indexLayerDiam(i);
        xminR(1,i)=indexLayerDiam(i);
    end
end

%% OPTIMIZATION PARAMETERS

V=6; % number of optimization variables
M=2+31; % number of optimization functions + most important features of 
        % each individual

% Upper and Lower bound variable range
dimax=[xmaxL,xmaxR];
dimin=[xminL,xminR];

txl=zeros(1,V)+dimin;
txu=zeros(1,V)+dimax;
xl=(txl(1,1:V));            % lower bound vector
xu=(txu(1,1:V));            % upper bound vectorfor 
etac = 20;                  % distribution index for crossover
etam = 20;                  % distribution index for mutation / mutation constant
          
pm=1/V;                     % Mutation Probability

Q=[];

%% Initial population 
xl_temp=repmat(xl, pop_size,1);
xu_temp=repmat(xu, pop_size,1);
x = xl_temp+((xu_temp-xl_temp).*rand(pop_size,V)); % x

for i =1:pop_size
    %% Design optimization
    % Optimization design
    db1l=rebarAvailable(1+fix(x(i,1)),2);
    db2l=rebarAvailable(1+fix(x(i,2)),2);
    db3l=rebarAvailable(1+fix(x(i,3)),2);

    db1m=rebarAvailable(1+fix(x(i,4)),2);
    db2m=rebarAvailable(1+fix(x(i,5)),2);
    db3m=rebarAvailable(1+fix(x(i,6)),2);

    [sepMin1l,sepMax1l]=sepMinMaxHK13([db1l],hagg,0);
    [sepMin2l,sepMax2l]=sepMinMaxHK13([db2l],hagg,0);
    [sepMin3l,sepMax3l]=sepMinMaxHK13([db3l],hagg,0);

    [sepMin1m,sepMax1m]=sepMinMaxHK13([db1m],hagg,0);
    [sepMin2m,sepMax2m]=sepMinMaxHK13([db2m],hagg,0);
    [sepMin3m,sepMax3m]=sepMinMaxHK13([db3m],hagg,0);

    sepMin=[sepMin1l,sepMin2l,sepMin3l,...
            sepMin1m,sepMin2m,sepMin3m];

    dbc=[db1l,db2l,db3l,db1m,db2m,db3m];
    abc = dbc.^2*pi/4;
    if all([dbc(1)>=dbc(2),dbc(2)>=dbc(3),...
             dbc(4)>=dbc(5),dbc(5)>=dbc(6)])

        rr1=b-2*brec-2*dvs;
        rr2=b-2*brec-2*dvs;
        rr3=b-2*brec-2*dvs;

        rr21=sepMin(1);
        rr22=sepMin(2);
        rr23=sepMin(3);

        nbmax1l=max([fix((rr1)/(rr21)),2]);
        nbmax2l=max([fix((rr2)/(rr22))]);
        nbmax3l=max([fix((rr3)/(rr23))]);

        nbmaxl=[nbmax1l,nbmax2l,nbmax3l];

        [nb1l,nb2l,nb3l,isfeasible]=nbSimple3D3L(nbmaxl,Ao3(1),abc(1:3),nbcc) ;
            
        rr1=b-2*brec-2*dvs;
        rr2=b-2*brec-2*dvs;
        rr3=b-2*brec-2*dvs;

        rr21=sepMin(4);
        rr22=sepMin(5);
        rr23=sepMin(6);

        nbmax1r=max([fix((rr1)/(rr21)),2]);
        nbmax2r=max([fix((rr2)/(rr22))]);
        nbmax3r=max([fix((rr3)/(rr23))]);
        nbmaxm=[nbmax1r,nbmax2r,nbmax3r];

        [nb1m,nb2m,nb3m,isfeasible]=nbSimple3D3L(nbmaxm,Ao3(2),abc(4:6),nblow);
        
        nblm=[nb1l,nb2l,nb3l,nb1m,nb2m,nb3m];

        [rebarVol(i),LenRebar{i},sepRebar(:,:,i),NbCombo9(i,:),EffLeft(i),EffMid(i),...
        EffRight(i),MrLeft(i),MrMid(i),MrRight(i),cLeft(i),cMid(i),cRight(i),ListRebarDiamLeft{i},...
        ListRebarDiamMid{i},ListRebarDiamRight{i},DistrRebarLeft{i},RebarDistrMid{i},...
        DistrRebarRight{i},nbcut3sec(:,:,i),nblowLeft(i,:),dblowLeft(i,:),nbTopMid(i,:),...
        dbTopMid(i,:),nblowRight(i,:),dblowRight(i,:),bestCFA(i),cc(i)]=CutRedistrOptimRecBeam(load_conditions,fcu,...
        Es,h,b,span,dbc,hagg,brec,hrec,pmin,pmax,sepMin,cutLoc,...
        Ao3,Wfac,nblm);

        dbc9(i,:)=[dbc,dbc(1:3)];
        db2c9(i,:)=[dblowLeft(i,:),dbTopMid(i,:),dblowRight(i,:)];
        nb2c9(i,:)=[nblowLeft(i,:),nbTopMid(i,:),nblowRight(i,:)];
    else
        rebarVol(i)=1e10;
        LenRebar{i}=0;
        NbCombo9(i,:)=zeros(1,9);
        sepRebar(:,:,i)=zeros(3,3);

        EffLeft(i)=0;
        EffMid(i)=0;
        EffRight(i)=0;

        MrLeft(i)=0;
        MrMid(i)=0;
        MrRight(i)=0;

        RebarDistrMid{i}=[];
        ListRebarDiamMid{i}=0;

        DistrRebarLeft{i}=[];
        ListRebarDiamLeft{i}=0;

        DistrRebarRight{i}=[];
        ListRebarDiamRight{i}=0;

        cLeft(i)=0;
        cMid(i)=0;
        cRight(i)=0;

        nblowLeft(i,:)=zeros(1,3);
        dblowLeft(i,:)=zeros(1,3);

        nbTopMid(i,:)=zeros(1,3);
        dbTopMid(i,:)=zeros(1,3);

        nblowRight(i,:)=zeros(1,3);
        dblowRight(i,:)=zeros(1,3);

        nbcut3sec(:,:,i)=zeros(3,3);

        dbc9(i,:)=zeros(1,9);

        db2c9(i,:)=zeros(1,9);
        nb2c9(i,:)=zeros(1,9);
        bestCFA(i)=0;
        cc(i)=10;
    end
    
    %% Construction cost
    bestEff(i,:)=[EffLeft(i),EffMid(i),EffRight(i)];
    bestMr(i,:)=[MrLeft(i),MrMid(i),MrRight(i)];
    
    puRecBeam=unitCostCardBeamsRec(pccb(1),pccb(2),pccb(3),...
                             pccb(4)*bestCFA(i),pccb(5),pccb(6),pccb(7));

    bestCost(i,1)=rebarVol(i)*wac*puRecBeam;

    %% Objective functions
    ff(i,:)=[rebarVol(i) 1-bestCFA(i)];
    
    %% Evlauation of constraints violation
    err(i,:)=(cc(i)>0).*cc(i);
end

ff2=[];
for i=1:pop_size
    ff2=[ff2;
        [ff(i,:), bestCost(i,1), NbCombo9(i,:), dbc9(i,:), cc(i)]];
end

error_norm=normalisation(err); % Normalisation of the constraint violation

optimal_features_ff=[bestCost NbCombo9 dbc9 nb2c9 bestEff];

population_init=[x ff optimal_features_ff error_norm];
[population front]=NDS_CD_cons(population_init,V,M);    % Non domination Sorting on initial population

IGDv=zeros(gen_max,1);
IGDt=zeros(gen_max,1);

t1=datetime;
%% Generation Starts
for gen_count=1:gen_max
    % selection (Parent Pt of 'N' pop size)
    parent_selected=tour_selection(population);                     % 10 Tournament selection
    %%% Reproduction (Offspring Qt of 'N' pop size)
    child_offspring  = genetic_operator(parent_selected(:,1:V),V,xl,xu,etac,etam,pm);    % SBX crossover and polynomial mutation

    for ii = 1:length(child_offspring(:,1))
        %% Design optimization
        % Optimization design
        db1l=rebarAvailable(fix(child_offspring(ii,1)),2);
        db2l=rebarAvailable(fix(child_offspring(ii,2)),2);
        db3l=rebarAvailable(fix(child_offspring(ii,3)),2);

        db1m=rebarAvailable(fix(child_offspring(ii,4)),2);
        db2m=rebarAvailable(fix(child_offspring(ii,5)),2);
        db3m=rebarAvailable(fix(child_offspring(ii,6)),2);

        [sepMin1l,sepMax1l]=sepMinMaxHK13([db1l],hagg,0);
        [sepMin2l,sepMax2l]=sepMinMaxHK13([db2l],hagg,0);
        [sepMin3l,sepMax3l]=sepMinMaxHK13([db3l],hagg,0);

        [sepMin1m,sepMax1m]=sepMinMaxHK13([db1m],hagg,0);
        [sepMin2m,sepMax2m]=sepMinMaxHK13([db2m],hagg,0);
        [sepMin3m,sepMax3m]=sepMinMaxHK13([db3m],hagg,0);

        sepMin=[sepMin1l,sepMin2l,sepMin3l,...
                sepMin1m,sepMin2m,sepMin3m];

        dbc=[db1l,db2l,db3l,db1m,db2m,db3m];
        if all([dbc(1)>=dbc(2),dbc(2)>=dbc(3),...
             dbc(4)>=dbc(5),dbc(5)>=dbc(6)])

            rr1=b-2*brec-2*dvs;
            rr2=b-2*brec-2*dvs;
            rr3=b-2*brec-2*dvs;

            rr21=sepMin(1);
            rr22=sepMin(2);
            rr23=sepMin(3);

            nbmax1l=max([fix((rr1)/(rr21)),2]);
            nbmax2l=max([fix((rr2)/(rr22))]);
            nbmax3l=max([fix((rr3)/(rr23))]);
            nbmaxl=[nbmax1l,nbmax2l,nbmax3l];

            [nb1l,nb2l,nb3l,isfeasible]=nbSimple3D3L(nbmaxl,Ao3(1),abc(1:3),nbcc) ;

            rr1=b-2*brec-2*dvs;
            rr2=b-2*brec-2*dvs;
            rr3=b-2*brec-2*dvs;

            rr21=sepMin(4);
            rr22=sepMin(5);
            rr23=sepMin(6);

            nbmax1r=max([fix((rr1)/(rr21)),2]);
            nbmax2r=max([fix((rr2)/(rr22))]);
            nbmax3r=max([fix((rr3)/(rr23))]);
            nbmaxm=[nbmax1r,nbmax2r,nbmax3r];

            [nb1m,nb2m,nb3m,isfeasible]=nbSimple3D3L(nbmaxm,Ao3(2),abc(4:6),nblow);

            nblm=[nb1l,nb2l,nb3l,nb1m,nb2m,nb3m];

            [rebarVol(ii),LenRebar{ii},sepRebar(:,:,ii),NbCombo9(ii,:),EffLeft(ii),EffMid(ii),...
            EffRight(ii),MrLeft(ii),MrMid(ii),MrRight(ii),cLeft(ii),cMid(ii),cRight(ii),ListRebarDiamLeft{ii},...
            ListRebarDiamMid{ii},ListRebarDiamRight{ii},DistrRebarLeft{ii},RebarDistrMid{ii},...
            DistrRebarRight{ii},nbcut3sec(:,:,ii),nblowLeft(ii,:),dblowLeft(ii,:),nbTopMid(ii,:),...
            dbTopMid(ii,:),nblowRight(ii,:),dblowRight(ii,:),bestCFA(ii),cc(ii)]=CutRedistrOptimRecBeam(load_conditions,fcu,...
            Es,h,b,span,dbc,hagg,brec,hrec,pmin,pmax,sepMin,cutLoc,...
            Ao3,Wfac,nblm);

            dbc9(ii,:)=[dbc,dbc(1:3)];
            db2c9(ii,:)=[dblowLeft(ii,:),dbTopMid(ii,:),dblowRight(ii,:)];
            nb2c9(ii,:)=[nblowLeft(ii,:),nbTopMid(ii,:),nblowRight(ii,:)];
        else
            rebarVol(ii)=1e10;
            LenRebar{ii}=0;

            sepRebar(:,:,ii)=zeros(3,3);

            EffLeft(ii)=0;
            EffMid(ii)=0;
            EffRight(ii)=0;

            MrLeft(ii)=0;
            MrMid(ii)=0;
            MrRight(ii)=0;

            RebarDistrMid{ii}=[];
            ListRebarDiamMid{ii}=0;

            DistrRebarLeft{ii}=[];
            ListRebarDiamLeft{ii}=0;

            DistrRebarRight{ii}=[];
            ListRebarDiamRight{ii}=0;

            cLeft(ii)=0;
            cMid(ii)=0;
            cRight(ii)=0;

            nblowLeft(ii,:)=zeros(1,3);
            dblowLeft(ii,:)=zeros(1,3);

            nbTopMid(ii,:)=zeros(1,3);
            dbTopMid(ii,:)=zeros(1,3);

            nblowRight(ii,:)=zeros(1,3);
            dblowRight(ii,:)=zeros(1,3);

            nbcut3sec(:,:,ii)=zeros(3,3);
                
            db2c9(ii,:)=zeros(1,9);
            nb2c9(ii,:)=zeros(1,9);
            dbc9(ii,:)=zeros(1,9);
            bestCFA(ii)=0;
            cc(ii)=10;
        end

        dbc9(ii,:)=[dbc,dbc(1:3)];
        db2c9(ii,:)=[dblowLeft(ii,:),dbTopMid(ii,:),dblowRight(ii,:)];
        nb2c9(ii,:)=[nblowLeft(ii,:),nbTopMid(ii,:),nblowRight(ii,:)];

        %% Construction cost
        bestEff(ii,:)=[EffLeft(ii),EffMid(ii),EffRight(ii)];
        bestMr(ii,:)=[MrLeft(ii),MrMid(ii),MrRight(ii)];

        puRecBeam=unitCostCardBeamsRec(pccb(1),pccb(2),pccb(3),...
                                 pccb(4)*bestCFA(ii),pccb(5),pccb(6),pccb(7));

        bestCost(ii,1)=rebarVol(ii)*wac*puRecBeam;
        
        %% Objective functions
        fff(ii,:)=[rebarVol(ii) 1-bestCFA(ii)];

        %% Evlauation of constraints violation
        err(ii,:)=(cc(ii)>0).*cc(ii);

    end
    for i=1:pop_size
        ff2=[ff2;
            [fff(i,:),bestCost(i,:),NbCombo9(i,:), dbc9(i,:), cc(i)]];
    end

    optimal_features_fff=[bestCost NbCombo9 dbc9 nb2c9 bestEff];

    error_norm=normalisation(err);                                
    child_offspring=[child_offspring fff optimal_features_fff error_norm];

    %%% INtermediate population (Rt= Pt U Qt of 2N size)
    population_inter=[population(:,1:V+M+1) ; child_offspring(:,1:V+M+1)];
    [population_inter_sorted front]=NDS_CD_cons(population_inter,V,M);              % Non domination Sorting on offspring
    %%% Replacement - N
    newPop=replacement(population_inter_sorted, front, pop_size);
    population=newPop;

    t2=datetime;
    tgen=seconds(t2-t1);

    genCFA(:,gen_count)=1-newPop(:,V+2);
    genVOL(:,gen_count)=newPop(:,V+1);

    %% Inverted Generational Distance

    MIGD=0;
    nonValid=0;
    for npi=1:pop_size
        xp=population(npi,V+2);
        yp=population(npi,V+1).*wac;
        minIGD=inf;
        for npj=1:np_baseline
            xp_base=1-CFA_PF_REF(npj);
            yp_base=W_PF_REF(npj);

            IGD=sqrt((xp-xp_base)^2+((yp-yp_base)/yp_max)^2);
            if IGD<minIGD
                minIGD=IGD;
            end
        end
        if minIGD>=1e5
            nonValid=nonValid+1;
        elseif minIGD<1e5
            MIGD=MIGD+minIGD;
        end
    end
    MIGD=MIGD/(pop_size-nonValid);

    if gen_count>1
        IGDv(gen_count,1)=(sum(IGDv(1:gen_count-1,1))+MIGD)/gen_count;
        dIGD=abs(IGDv(gen_count,1)-IGDv(gen_count-1,1));
    else
        IGDv(gen_count,1)=MIGD;
    end
    IGDt(gen_count,1)=tgen;

    if MIGDconv==1
        if gen_count>1
            if dIGD/max(IGDv)<0.01/100
                break;
            end
        end
    end
end
newPop=sortrows(newPop,V+1);

%% Collecting designs along the PF
cscore=newPop(:,8);
bestnb9=newPop(:,10:18);
bestcombdi9=newPop(:,19:27);
for i=1:pop_size
    % Principal optimization faces
    bestcombAbi9=bestcombdi9(i,:).^2*pi/4;

    bestAbTL1=bestnb9(i,1).*bestcombAbi9(1);
    bestAbTL2=bestnb9(i,2).*bestcombAbi9(2);
    bestAbTL3=bestnb9(i,3).*bestcombAbi9(3);

    bestAbBM1=bestnb9(i,4).*bestcombAbi9(4);
    bestAbBM2=bestnb9(i,5).*bestcombAbi9(5);
    bestAbBM3=bestnb9(i,6).*bestcombAbi9(6);

    bestAbTR1=bestnb9(i,7).*bestcombAbi9(7);
    bestAbTR2=bestnb9(i,8).*bestcombAbi9(8);
    bestAbTR3=bestnb9(i,9).*bestcombAbi9(9);

    best9Ab(i,:)=[bestAbTL1,bestAbTL2,bestAbTL3,...
                   bestAbBM1,bestAbBM2,bestAbBM3,...
                   bestAbTR1,bestAbTR2,bestAbTR3];

end
min2index=fix(1/4*pop_size);
midindex=fix(pop_size/2);
mid2index=fix(3/4*pop_size);

minOpt=best9Ab(1,:);
min2Opt=best9Ab(min2index,:);
midOpt=best9Ab(midindex,:);
mid2Opt=best9Ab(mid2index,:);
maxOpt=best9Ab(pop_size,:);

extrOptPFArea=[minOpt,min2Opt,midOpt,mid2Opt,maxOpt];

minOpt=cscore(1,:);
min2Opt=cscore(min2index,:);
midOpt=cscore(midindex,:);
mid2Opt=cscore(mid2index,:);
maxOpt=cscore(pop_size,:);

extrOptPFCS=[minOpt,min2Opt,midOpt,mid2Opt,maxOpt];

[PF_CFA,PF_VOL]=filterSingle(newPop(:,V+2),newPop(:,V+1));
PF_CFA=1-PF_CFA;

%% Results
% PARETO FRONT COST-CFA

if pltPF==1 && sum(ff2(:,22)==0)>0
    figure(1)
    xlabel('Constructability Score (CS)')
    ylabel('Rebar Weight (Kg)')
    title({'Pareto Front ``Rebar weight - CS´´','Rectangular Beam'})
    hold on
    plot(PF_CFA,PF_VOL.*wac,'m o','linewidth',1.5,...
        'MarkerFaceColor','magenta')
    hold on
    plot(CFA_PF_REF,W_PF_REF,'b -o','linewidth',1.5,...
        'MarkerFaceColor','blue')
    legend('Obtained PF','Reference PF')
    hold on
    set(gca, 'Fontname', 'Times New Roman','FontSize',15);
    
end

if sum(ff2(:,22)==0)==0
    disp('No solution was found')
end
end

%% Function appendix
function [parent_selected] = tour_selection(pool)
%%% Description

% 1. Parents are selected from the population pool for reproduction by using binary tournament selection
%    based on the rank and crowding distance. 
% 2. An individual is selected if the rank is lesser than the other or if
%    crowding distance is greater than the other.
% 3. Input and output are of same size [pop_size, V+M+3].


%%% Binary Tournament Selection
[pop_size, distance]=size(pool);
rank=distance-1;
candidate=[randperm(pop_size);randperm(pop_size)]';

for i = 1: pop_size
    parent=candidate(i,:);                                  % Two parents indexes are randomly selected
    if pool(parent(1),rank)~=pool(parent(2),rank)              % For parents with different ranks
        if pool(parent(1),rank)<pool(parent(2),rank)            % Checking the rank of two individuals
            mincandidate=pool(parent(1),:);
        elseif pool(parent(1),rank)>pool(parent(2),rank)
            mincandidate=pool(parent(2),:);
        end
        parent_selected(i,:)=mincandidate;                          % Minimum rank individual is selected finally
    else                                                       % for parents with same ranks  
        if pool(parent(1),distance)>pool(parent(2),distance)    % Checking the distance of two parents
            maxcandidate=pool(parent(1),:);
        elseif pool(parent(1),distance)< pool(parent(2),distance)
            maxcandidate=pool(parent(2),:);
        else
            temp=randperm(2);
            maxcandidate=pool(parent(temp(1)),:);
        end 
        parent_selected(i,:)=maxcandidate;                          % Maximum distance individual is selected finally
    end
end
end

function new_pop=replacement(population_inter_sorted, front, pop_size)

%% Description
% The next generation population is formed by appending each front subsequently until the
% population size exceeds the current population size. If When adding all the individuals
% of any front, the population exceeds the population size, then the required number of 
% remaining individuals alone are selected from that particular front based
% on crowding distance.
%% code starts
index=0;
ii=1;
while index < pop_size
    l_f=length(front(ii).fr);
    if index+l_f < pop_size 
        new_pop(index+1:index+l_f,:)= population_inter_sorted(index+1:index+l_f,:);
        index=index+l_f;
    else
            temp1=population_inter_sorted(index+1:index+l_f,:);
            temp2=sortrows(temp1,size(temp1,2));
            new_pop(index+1:pop_size,:)= temp2(l_f-(pop_size-index)+1:l_f,:);
            index=index+l_f;
    end
    ii=ii+1;
end
end

function mutated_child = poly_mutation(y,V, xl, xu, etam, pm)

%% Description
% 1. Input is the crossovered child of size (1,V) in the vector 'y' from 
%    'genetic_operator.m'.
% 2. Output is in the vector 'mutated_child' of size (1,V).
%% Polynomial mutation including boundary constraint
del=min((y-xl),(xu-y))./(xu-xl);
t=rand(1,V);
loc_mut=t<pm;        
u=rand(1,V);
delq=(u<=0.5).*((((2*u)+((1-2*u).*((1-del).^(etam+1)))).^(1/(etam+1)))-1)+...
    (u>0.5).*(1-((2*(1-u))+(2*(u-0.5).*((1-del).^(etam+1)))).^(1/(etam+1)));
c=y+delq.*loc_mut.*(xu-xl);
mutated_child=c;
end

function err_norm  = normalisation(error_pop)
  
%% Description
% 1. This function normalises the constraint violation of various 
%    individuals, since the range of constraint violation of every 
%    chromosome is not uniform.
% 2. Input is in the matrix error_pop with size [pop_size, number of 
%    constraints].
% 3. Output is a normalised vector, err_norm of size [pop_size,1]
 
%% Error Nomalisation
[N,nc]=size(error_pop);
con_max=0.001+max(error_pop);
con_maxx=repmat(con_max,N,1);
cc=error_pop./con_maxx;
err_norm=sum(cc,2);                % finally sum up all violations
end


function [chromosome_NDS_CD front] = NDS_CD_cons(population,V,M) 

%%%%% Description
% 1. This function is to perform Deb's fast elitist non-domination sorting and crowding distance assignment. 
% 2. Input is in the variable 'population' with size: [size(popuation), V+M+1]
% 3. This function returns 'chromosome_NDS_CD' with size [size(population),V+M+3]
% 4. A flag 'problem_type' is used to identify whether the population is fully feasible (problem_type=0) or fully infeasible (problem_type=1) 
%    or partly feasible (problem_type=0.5). 

%%% Reference:
%Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan, " A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II", 
%IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 6, No. 2, APRIL 2002. 


%%% function begins
% Initialising structures and variables
chromosome_NDS_CD1=[];
infpop=[];
front.fr=[];
struct.sp=[];
rank=1;


%%% Segregating feasible and infeasible solutions

if all(population(:,V+M+1)==0)
    problem_type=0;
    chromosome=population(:,1:V+M);                         % All Feasible chromosomes;    
    pop_size1=size(chromosome,1);
elseif all(population(:,V+M+1)~=0)
    problem_type=1;
    pop_size1=0;
    infchromosome=population;                               % All InFeasible chromosomes;       
else
    problem_type=0.5;
    feas_index=find(population(:,V+M+1)==0);
    chromosome=population(feas_index,1:V+M);                % Feasible chromosomes;    
    pop_size1=size(chromosome,1);
    infeas_index=find(population(:,V+M+1)~=0);
    infchromosome=population(infeas_index,1:V+M+1);         % infeasible chromosomes;    
end

%%% Handling feasible solutions 
if problem_type==0 || problem_type==0.5
    pop_size1 = size(chromosome,1);
    f1 = chromosome(:,V+1);   % objective function values
    f2 = chromosome(:,V+2);
    %Non- Domination Sorting 
    % First front
    for p=1:pop_size1
        struct(p).sp=find(((f1(p)-f1)<0 &(f2(p)-f2)<0) | ((f2(p)-f2)==0 &(f1(p)-f1)<0) | ((f1(p)-f1)==0 &(f2(p)-f2)<0)); 
        n(p)=length(find(((f1(p)-f1)>0 &(f2(p)-f2)>0) | ((f2(p)-f2)==0 &(f1(p)-f1)>0) | ((f1(p)-f1)==0 &(f2(p)-f2)>0)));
    end

    front(1).fr=find(n==0);
    % Creating subsequent fronts
    while (~isempty(front(rank).fr))

        front_indiv=front(rank).fr;
        n(front_indiv)=inf;
        chromosome(front_indiv,V+M+1)=rank;
        rank=rank+1;
        front(rank).fr=[];
        for i = 1:length(front_indiv)

            temp=struct(front_indiv(i)).sp;
            n(temp)=n(temp)-1;
        end 
        q=find(n==0);
        front(rank).fr=[front(rank).fr q];

    end
    chromosome_sorted=sortrows(chromosome,V+M+1);    % Ranked population  

    %Crowding distance Assignment
    rowsindex=1;
    for i = 1:length(front)-1
        l_f=length(front(i).fr);

        if l_f > 2

            sorted_indf1=[];
            sorted_indf2=[];
            sortedf1=[];
            sortedf2=[];
              % sorting based on f1 and f2;
            [sortedf1 sorted_indf1]=sortrows(chromosome_sorted(rowsindex:(rowsindex+l_f-1),V+1));
            [sortedf2 sorted_indf2]=sortrows(chromosome_sorted(rowsindex:(rowsindex+l_f-1),V+2));

            f1min=chromosome_sorted(sorted_indf1(1)+rowsindex-1,V+1);
            f1max=chromosome_sorted(sorted_indf1(end)+rowsindex-1,V+1);

            chromosome_sorted(sorted_indf1(1)+rowsindex-1,V+M+2)=inf;
            chromosome_sorted(sorted_indf1(end)+rowsindex-1,V+M+2)=inf;

            f2min=chromosome_sorted(sorted_indf2(1)+rowsindex-1,V+2);
            f2max=chromosome_sorted(sorted_indf2(end)+rowsindex-1,V+2);

            chromosome_sorted(sorted_indf2(1)+rowsindex-1,V+M+3)=inf;
            chromosome_sorted(sorted_indf2(end)+rowsindex-1,V+M+3)=inf;

            for j = 2:length(front(i).fr)-1

                if  (f1max - f1min == 0) || (f2max - f2min == 0)

                    chromosome_sorted(sorted_indf1(j)+rowsindex-1,V+M+2)=inf;
                    chromosome_sorted(sorted_indf2(j)+rowsindex-1,V+M+3)=inf;
                else
                    chromosome_sorted(sorted_indf1(j)+rowsindex-1,V+M+2)=(chromosome_sorted(sorted_indf1(j+1)+rowsindex-1,V+1)-chromosome_sorted(sorted_indf1(j-1)+rowsindex-1,V+1))/(f1max-f1min);
                    chromosome_sorted(sorted_indf2(j)+rowsindex-1,V+M+3)=(chromosome_sorted(sorted_indf2(j+1)+rowsindex-1,V+2)-chromosome_sorted(sorted_indf2(j-1)+rowsindex-1,V+2))/(f2max-f2min);
                end
            end


        else

            chromosome_sorted(rowsindex:(rowsindex+l_f-1),V+M+2:V+M+3)=inf;
        end
        rowsindex = rowsindex + l_f;
    end
    chromosome_sorted(:,V+M+4) = sum(chromosome_sorted(:,V+M+2:V+M+3),2); 
    chromosome_NDS_CD1 = [chromosome_sorted(:,1:V+M) zeros(pop_size1,1) chromosome_sorted(:,V+M+1) chromosome_sorted(:,V+M+4)]; % Final Output Variable

end

%%% Handling infeasible solutions
if problem_type==1 | problem_type==0.5
    infpop=sortrows(infchromosome,V+M+1);
    infpop=[infpop(:,1:V+M+1) (rank:rank-1+size(infpop,1))' inf*(ones(size(infpop,1),1))];
    for kk = (size(front,2)):(size(front,2))+(length(infchromosome))-1
        front(kk).fr= pop_size1+1;
    end
end
chromosome_NDS_CD = [chromosome_NDS_CD1;infpop]; 
end

function child_offspring  = genetic_operator(parent_selected,V,xl,xu,etac,etam,pm)

%% Description
% 1. Crossover followed by mutation
% 2. Input is in 'parent_selected' matrix of size [pop_size,V].
% 3. Output is also of same size in 'child_offspring'. 

%% Reference 
% Deb & samir agrawal,"A Niched-Penalty Approach for Constraint Handling in Genetic Algorithms". 
%% SBX cross over operation incorporating boundary constraint
[N] = size(parent_selected,1);
xl1=xl';
xu1=xu';
rc=randperm(N);
for i=1:(N/2)
    parent1=parent_selected((rc(2*i-1)),:);
    parent2=parent_selected((rc(2*i)),:);
    if (isequal(parent1,parent2))==1 & rand(1)>0.5
        child1=parent1;
        child2=parent2;
    else 
        for j = 1: V  
            if parent1(j)<parent2(j)
               beta(j)= 1 + (2/(parent2(j)-parent1(j)))*(min((parent1(j)-xl1(j)),(xu1(j)-parent2(j))));
            else
               beta(j)= 1 + (2/(parent1(j)-parent2(j)))*(min((parent2(j)-xl1(j)),(xu1(j)-parent1(j))));
            end   
        end
         u=rand(1,V);
         alpha=2-beta.^-(etac+1);
         betaq=(u<=(1./alpha)).*(u.*alpha).^(1/(etac+1))+(u>(1./alpha)).*(1./(2 - u.*alpha)).^(1/(etac+1));
        child1=0.5*(((1 + betaq).*parent1) + (1 - betaq).*parent2);
        child2=0.5*(((1 - betaq).*parent1) + (1 + betaq).*parent2);
    end
    child_offspring((rc(2*i-1)),:)=poly_mutation(child1,V, xl, xu, etam, pm);           % polynomial mutation
    child_offspring((rc(2*i)),:)=poly_mutation(child2,V, xl, xu, etam, pm);             % polynomial mutation
end
end


%% Function appendix

%% Function appendix
function [nb1,nb2,nb3,isfeasible]=nbSimple3D3L(nbmax,Aos,ab,nbcc)
    
    Abmax1 = nbmax(1) * ab(1) ;
    Abmax2 = nbmax(2) * ab(2) ;
    Abmax3 = nbmax(3) * ab(3) ;
    
    if sum(nbcc.*ab)>=Aos
        isfeasible=true;
        nb1=nbcc(1);
        nb2=nbcc(2);
        nb3=nbcc(3);
    else
        if Abmax1 >= Aos % only one rebar layer is necessary
 
            isfeasible=true;
            
            if any([nbcc(1)==2,nbcc(1)==0]) 
                nb1 = min([ceil(Aos / ab(1)),nbmax(1)]) ;
                if nb1<2
                    nb1=2;
                end
            elseif nbcc(1)>2
                nb1 = nbcc(1) ;
            end
            
            Ab1=nb1*ab(1);
            if Aos-Ab1>0
                nb2=ceil((Aos-Ab1) / ab(2));
                if nb2>nb1
                    nb2=nb1;
                end
                Ab2=nb2*ab(2);
                if Aos-Ab1-Ab2 > 0
                    nb3=ceil((Aos-Ab1-Ab2) / ab(3));
                end
                if nb3>nb2
                    nb3=nb2;
                end
            else
                nb2=max([0,nbcc(2)]);
                nb3=max([0,nbcc(3)]);
            end

        elseif Abmax1 + Abmax2 >= Aos % only two rebar layers are necessary

            isfeasible=true;
            
            if any([nbcc(1)==2,nbcc(1)==0]) 
                nb1= min(ceil(Aos / ab(1)),nbmax(1)) ;
                if nb1<2
                    nb1=2;
                end
            elseif nbcc(1)>2
                nb1=nbcc(1);
            end
            
            Ab1=nb1*ab(1);
            if any([nbcc(2)==0,nbcc(2)==2])
                nb2=min([ceil((Aos-Ab1)/ab(2)),nbmax(2)]);
                if nb2>nb1
                    nb2=nb1;
                end
                Ab2=nb2*ab(2);
                if Aos - Ab1 - Ab2 > 0
                    nb3=min([ceil((Aos-Ab1-Ab2)/ab(3)),nbmax(3)]);
                else
                    nb3=nbcc(3);
                end
            elseif nbcc(2)==1
                nb2=min([ceil((Aos-Ab1)/ab(2)),nbmax(2)]);
                if mod(nb2,2)==0
                    if any([nb2==nbmax(2),nb2==nb1]) 
                        nb2=nb2-1;
                        if nbcc(3)==0
                            nb3=1;
                        else
                            nb3=nbcc(3);
                        end
                    else
                        nb2=nb2+1;
                        nb3=nbcc(3);
                    end
                else
                    nb3=nbcc(3);
                end
            elseif nbcc(2)>2
                nb2=nbcc(2);
                Ab2=nb2*ab(2);
                if Aos - Ab1 - Ab2 > 0
                    nb3=min([ceil((Aos-Ab1-Ab2)/ab(3)),nbmax(3)]);
                    if nb3>nb2
                        nb3=nb2;
                    end
                else
                    nb3=nbcc(3);
                end
            end

            if mod(nb1,2)==0 && mod(nb2,2)~=0
                if any([nb2==nbmax(2),nb2==nb1])
                    nb2=nb2-1;
                    nb3=nb3+1;
                else
                    nb2=nb2+1;
                end
            end
            if mod(nb2,2)==0 && mod(nb3,2)~=0
                nb3=nb3+1;
            end

        elseif Abmax1 + Abmax2 + Abmax3 >= Aos

            isfeasible=true;
            if any([nbcc(1)==2,nbcc(1)==0]) 
                nb1= min([ceil(Aos / ab(1)),nbmax(1)]) ;
                if nb1<2
                    nb1=2;
                end
            elseif nbcc(1)>2
                nb1=nbcc(1);
            end
            
            Ab1=nb1*ab(1);
            if any([nbcc(2)==0,nbcc(2)==2])
                nb2=min(ceil((Aos-Ab1)/ab(2)),nbmax(2));
                if nb2>nb1
                    nb2=nb1;
                end
            elseif nbcc(2)==1
                nb2=min(ceil((Aos-Ab1)/ab(2)),nbmax(2));
                if nb2>nb1
                    nb2=nb1;
                end
                if mod(nb2,2)==0
                    if any([nb2==nbmax(2),nb2==nb1])
                        nb2=nb2-1;
                        if nbcc(3)==0
                            nb3=1;
                        else
                            nb3=nbcc(3);
                        end
                    else
                        nb2=nb2+1;
                        nb3=nbcc(3);
                    end
                else
                    nb3=nbcc(3);
                end
            elseif nbcc(2)>2
                nb2=nbcc(2);
                nb3=nbcc(3);
            end
            
            Ab2=nb2*ab(2);
            if any([nbcc(3)==0,nbcc(3)==2])
                nb3 = min([ceil((Aos-Ab1-Ab2)/ab(3)),nbmax(3)]);
                if nb3>nb2
                    nb3=nb2;
                end
            elseif nbcc(3)==1
                nb3=ceil(min((Aos-Ab1-Ab2)/ab(3),nbmax(3)));
                if mod(nb3,2)==0
                    if nb3==nbmax(2)
                        nb3=nb3-1;
                    else
                        nb3=nb3+1;
                    end
                end
            elseif nbcc(3)>2
                nb3=nbcc(3);
            end
    
            if mod(nb1,2)==0 && mod(nb2,2)~=0
                nb2=nb2+1;
            end
    
            if mod(nb2,2)==0 && mod(nb3,2)~=0
                nb3=nb3+1;
            end
        else
            nb1 = 0;
            nb2 = 0;
            nb3 = 0;
            isfeasible=false;
        end
    end
end
