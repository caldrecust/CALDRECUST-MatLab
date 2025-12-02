function [minimumFitness,bestLenRebar,bestsepRebar,bestPosition,bestdbc18,...
    bestEfL,bestEffMid,bestEffRight,bestMrLeft,bestMrMid,bestMrRight,cbestLeft,...
    cBestMid,cBestRight,bestListRebarDiamLeft,bestListRebarDiamMid,...
    bestListRebarDiamRight,bestDistrRebarLeft,bestRebarDistrMid,...
    bestDistrRebarRight,bestnbtLMR,bestnbcut3sec,bestnblowRight,bestdblowRight,...
    bestCFA]=GABeamsRebarContSpan2(b,h,span,brec,hrec,hagg,pmin,pmax,rebarAvailable,...
    fcu,load_conditions,fy,wac,cutLoc,dbcc,nbcc,dblow,nblow,Wfac,graphConvPlot,ispan)

%------------------------------------------------------------------------
% Syntax:
% [c_best,bestMr,bestEf,best_area,tbest,h]=PSO3layerBeamsRebar3sec(b,h,duct,...
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
%                               (Kg/cm2)
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2023-07-03
% Copyright (c)  School of Engineering
%                HKUST
%------------------------------------------------------------------------
    Es=fy/0.00217;

    % Max and min reinforcement area
    ndiam=length(rebarAvailable(:,1));

    % Identifying existing rebar diameters in left section

    indexLayerDiam=[];
    for j=1:6
        for i=1:length(rebarAvailable(:,1))
            if dbcc(j)==rebarAvailable(i,2)
                indexLayerDiam=[indexLayerDiam,i];
            end
        end
    end

    % Identifying layers of rebar valid for optimization for left section

    indexLayerOptim=zeros(1,3);
    xmaxL=zeros(1,6);
    xminL=zeros(1,6);
    for i=1:6
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
    for j=1:6
        for i=1:length(rebarAvailable(:,1))
            if dblow(j)==rebarAvailable(i,2)
                indexLayerDiam=[indexLayerDiam,i];
            end
        end
    end

    % Identifying layers of rebar valid for optimization for right section

    indexLayerOptim=zeros(1,6);
    xmaxR=zeros(1,6);
    xminR=zeros(1,6);
    for i=1:6
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

    %% Number of rebars
    dbmin=rebarAvailable(1,2);

    [sepMindbmin,sepMax1m]=sepMinMaxHK13(dbmin,hagg,0);
    nbmax=fix((b-2*brec-2*10-2*sepMindbmin+sepMindbmin)/(dbmin+sepMindbmin));

    xmax=[xmaxL,xmaxR,nbmax,  2,nbmax,  2,nbmax,  nbmax,   2,nbmax,   2,nbmax];
    xmin=[xminL,xminR,nbcc(2),0,nbcc(4),0,nbcc(6),nblow(2),0,nblow(4),0,nblow(6)];

    variableRange=[xmin',xmax'];

    %% GA algorithmic parameters
    crossoverProbability=0.65;
    mutationProbability=0.06;
    tournamentSelectionParameter=0.8;
    populationSize=150;
    numberOfGenes=80;

    numberVariables=22;
    numberCopies=1;
    tournamentSize=2;

    %% Initialization of generations
    decodedPopulation=zeros(populationSize,numberVariables);
    population=InitializePopulation(populationSize,numberOfGenes);
    fitness=zeros(populationSize,1);

    %% Main loop
    bestIndividualIndex=0; 
    cbestLeft=[];
    iter=1;
    nGen=300;
    minimumFitness=inf; % Assumes non-negative fitness values!
    while iter<=nGen
        for j=1:populationSize
            chromosome=population(j,:);

            x=DecodeChromosome(chromosome,variableRange);
            decodedPopulation(j,:)=x;

            db1l=rebarAvailable(x(1),2);
            db2l=rebarAvailable(x(2),2);
            db3l=rebarAvailable(x(3),2);
            db4l=rebarAvailable(x(4),2);
            db5l=rebarAvailable(x(5),2);
            db6l=rebarAvailable(x(6),2);

            db1m=rebarAvailable(x(7),2);
            db2m=rebarAvailable(x(8),2);
            db3m=rebarAvailable(x(9),2);
            db4m=rebarAvailable(x(10),2);
            db5m=rebarAvailable(x(11),2);
            db6m=rebarAvailable(x(12),2);

            [sepMin1l,sepMax1l]=sepMinMaxHK13([db1l,db2l],hagg,0);
            [sepMin2l,sepMax2l]=sepMinMaxHK13([db3l,db4l],hagg,0);
            [sepMin3l,sepMax3l]=sepMinMaxHK13([db5l,db6l],hagg,0);

            [sepMin1m,sepMax1m]=sepMinMaxHK13([db1m,db2m],hagg,0);
            [sepMin2m,sepMax2m]=sepMinMaxHK13([db3m,db4m],hagg,0);
            [sepMin3m,sepMax3m]=sepMinMaxHK13([db5m,db6m],hagg,0);

            sepMin=[sepMin1l,sepMin2l,sepMin3l,...
                    sepMin1m,sepMin2m,sepMin3m];

            dbc=[db1l,db2l,db3l,db4l,db5l,db6l,db1m,db2m,db3m,db4m,db5m,db6m];

            nb1l=2;
            nb2l=x(13);
            nb3l=x(14);
            nb4l=x(15);
            nb5l=x(16);
            nb6l=x(17);

            nb1m=2;
            nb2m=x(18);
            nb3m=x(19);
            nb4m=x(20);
            nb5m=x(21);
            nb6m=x(22);

            nblm=[nb1l,nb2l,nb3l,nb4l,nb5l,nb6l,nb1m,nb2m,nb3m,nb4m,nb5m,nb6m];

            [fitness(i),LenRebar,sepRebar,NbCombo18,EffLeft,EffMid,...
            EffRight,MrLeft,MrMid,MrRight,cLeft,cMid,cRight,ListRebarDiamLeft,...
            ListRebarDiamMid,ListRebarDiamRight,DistrRebarLeft,RebarDistrMid,...
            DistrRebarRight,dbcRight,nbcut3sec,nblowLeft,dblowLeft,nbTopMid,...
            dbTopMid,nblowRight,dblowRight,CFA,const]=CutRedistrOptimRecBeam(load_conditions,fcu,...
            Es,h,b,span,dbc,hagg,brec,hrec,pmin,pmax,sepMin,rebarAvailable,cutLoc,...
            Wfac,nblm);

            dbc18=[dbc,dbcRight];

            if (fitness(i)<minimumFitness && const==0)
                bestIndividualIndex=i;
                minimumFitness=fitness(i); % rebar volume (cm^3)
                bestCFA=CFA;
                bestLenRebar=LenRebar;
                bestsepRebar=sepRebar;
                bestPosition=x;
                bestdbc18=dbc18;
                bestEfL=EffLeft;
                bestEffMid=EffMid;
                bestEffRight=EffRight;
                bestMrLeft=MrLeft;
                bestMrMid=MrMid;
                bestMrRight=MrRight;
                cbestLeft=cLeft;
                cBestMid=cMid;
                cBestRight=cRight;
                bestListRebarDiamLeft=ListRebarDiamLeft;
                bestListRebarDiamMid=ListRebarDiamMid;
                bestListRebarDiamRight=ListRebarDiamRight;
                bestDistrRebarLeft=DistrRebarLeft;
                bestRebarDistrMid=RebarDistrMid;
                bestDistrRebarRight=DistrRebarRight;
                bestnbtLMR=NbCombo18;
                bestnbcut3sec=nbcut3sec;
                bestnblowRight=nblowRight;
                bestdblowRight=dblowRight;
                bestnblowLeft=nblowLeft;
                bestdblowLeft=dblowLeft;
                bestnbTopMid=nbTopMid;
                bestdbTopMid=dbTopMid;
            end
        end
        fitnessHist(iter,1)=minimumFitness;
        if bestIndividualIndex==0
            bestIndividualIndex=ceil(rand*nGen);
        end
        tempPopulation=population;

        for i=1:2:populationSize
            i1=TournamentSelect(fitness,tournamentSelectionParameter,tournamentSize);
            i2=TournamentSelect(fitness,tournamentSelectionParameter,tournamentSize);
            chromosome1=population(i1,:);
            chromosome2=population(i2,:);

            k=rand;
            if(k<crossoverProbability)
                newChromosomePair=Cross(chromosome1,chromosome2);
                tempPopulation(i,:)=newChromosomePair(1,:);
                tempPopulation(i+1,:)=newChromosomePair(2,:);
            else
                tempPopulation(i,:)=chromosome1;
                tempPopulation(i+1,:)=chromosome2;
            end
        end 

        for i=1:populationSize
            originalChromosome=tempPopulation(i,:);
            mutatedChromosome=Mutate(originalChromosome,mutationProbability);
            tempPopulation(i,:)=mutatedChromosome;
        end

        bestIndividual=InsertBestIndividual(population,...
            bestIndividualIndex,numberCopies);

        for i=1:numberCopies    
            tempPopulation(i,:)=bestIndividual(i,:);
        end

        population=tempPopulation;
        iter=iter+1;
    end

    % Optima Convergence Graph
    if graphConvPlot==1 && isempty(cbestLeft)==0
        figure(4)
        if sum(nbcc)==0
            plot(1:1:nGen,...
               fitnessHist.*wac,'-o')
            hold on
            legend('Span 1')
        else
            plot(1:1:nGen,...
               fitnessHist.*wac,'-o','DisplayName',strcat('Span',num2str(ispan)))

            hold on
        end
        xlabel('Iteration')
        ylabel('Rebar weight (N)')
        title({'Optimum convergence of reinforcement weight for a concrete beam';
                'GA'})
        hold on
    else
        minimumFitness=1e10; % rebar volume (mm^3)
        bestLenRebar=0;
        bestCFA=0;
        bestsepRebar=zeros(3,3);
        bestPosition=[];
        bestdbc18=zeros(1,18);
        bestEfL=0;
        bestEffMid=0;
        bestEffRight=0;
        bestMrLeft=0;
        bestMrMid=0;
        bestMrRight=0;
        cbestLeft=0;
        cBestMid=0;
        cBestRight=0;
        bestListRebarDiamLeft=[];
        bestListRebarDiamMid=[];
        bestListRebarDiamRight=[];
        bestDistrRebarLeft=[];
        bestRebarDistrMid=[];
        bestDistrRebarRight=[];
        bestnbtLMR=zeros(1,18);
        bestnbcut3sec=[];

        bestnblowRight=zeros(1,6);
        bestdblowRight=zeros(1,6);
        bestnblowLeft=zeros(1,6);
        bestdblowLeft=zeros(1,6);
        bestnbTopMid=zeros(1,6);
        bestdbTopMid=zeros(1,6);
    end
end


%% Function appendix for this optimizaiton process
function population=InitializePopulation(populationSize,nGenes)

    population=zeros(populationSize,nGenes);
    for i=1:populationSize
        for j=1:nGenes
            s=rand;
            if (s<0.5)
                population(i,j)=0;
            else
                population(i,j)=1;
            end
        end
    end
end

function bestIndividual=InsertBestIndividual(population,...
                        bestIndividualIndex,numberCopies)

    nGenes=size(population,2);
    bestIndividual=zeros(numberCopies,nGenes);
    for i=1:numberCopies
        bestIndividual(i,:)=population(bestIndividualIndex,:);
    
    end
end

function mutatedChromosome=Mutate(chromosome,mutationProbability)

    nGenes=size(chromosome,2);
    mutatedChromosome=chromosome;
    for j=1:nGenes
        r=rand;
        if (r<mutationProbability)
            mutatedChromosome(j)=1-chromosome(j);
        end
    end
end

function newChromosomePair=Cross(chromosome1,chromosome2)

    nGenes=size(chromosome1,2); %Both chromosomes must have the same length
    
    crossoverPoint=1+fix(rand*(nGenes-1));
    
    newChromosomePair=zeros(2,nGenes);
    for j=1:nGenes
        if (j<=crossoverPoint)
            newChromosomePair(1,j)=chromosome1(j);
            newChromosomePair(2,j)=chromosome2(j);
        else
           newChromosomePair(1,j)=chromosome2(j);
           newChromosomePair(2,j)=chromosome1(j); 
        end
    end
end

function [dnb3]=DecodeChromosome(chromosome,variableRange)

    numberVariables=length(variableRange(:,1));
    
    nGenes=size(chromosome,2);
    k=fix(nGenes/numberVariables);
    dnb3=zeros(1,numberVariables);
    x=zeros(1,numberVariables);
    for i=1:numberVariables
        x(i)=0.0;
        for j=1:k
            x(i)=x(i)+chromosome(j+(i-1)*k)*2^(-j);
        end
        
        dnb3(i)=fix(variableRange(i,1)+(variableRange(i,2)-...
               variableRange(i,1))*x(i)/(1-2^(-k)));
            
    end
end

function iSelected=TournamentSelect(fitness,pTournament,tournamentSize)
    
    populationSize=size(fitness,1);
    iTmp=zeros(tournamentSize);
    
    r=rand;
    
    for j=1:tournamentSize
        iTmp(j)=1+fix(rand*populationSize);
    end
    
    if (r<pTournament)
        maxFitness=fitness(iTmp(1));
        iSelected=iTmp(1);
        for j=1:tournamentSize-1
            
            if (maxFitness<fitness(iTmp(j+1)))
                maxFitness=fitness(iTmp(j+1));
                iSelected=iTmp(j+1);
            end
        end
    
    else
        minFitness=fitness(iTmp(1));
        iSelected=iTmp(1);
        for j=1:tournamentSize-1
            
            if (minFitness>fitness(iTmp(j+1)))
                minFitness=fitness(iTmp(j+1));
                iSelected=iTmp(j+1);
            end
        end

    end
end