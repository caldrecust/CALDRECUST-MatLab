function [bestsepRebar,bestdbc8,bestEfL,bestMrLeft,cbestLeft,...
    bestListRebarDiamLeft,bestDistrRebarLeft,bestnbt]=PSOptim1secBeamsRebar(b,h,...
    brec,hrec,hagg,pmin,pmax,rebarAvailable,fcu,Mul,fy,wac,...
    graphConvPlot,ispan,layPlt)

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

amin=pmin*b*h;
amax=pmax*b*h;

% Max and min reinforcement area
ndiam=length(rebarAvailable(:,1));

% Identifying layers of rebar valid for optimization for left section

xmaxDiam=zeros(1,8)+ndiam;
xminDiam=zeros(1,8)+1;

dbmin=rebarAvailable(1,2);
[sepMindbmin,sepMax1m]=sepMinMaxHK13(dbmin,hagg,0);
nbmax=fix((b-2*brec-2*10-2*sepMindbmin+sepMindbmin)/(dbmin+sepMindbmin));

xmaxNb=[nbmax,2,nbmax,2,nbmax,nbmax];
xminNb=[0,0,0,0,0,0];

xmax=[xmaxDiam,xmaxNb];
xmin=[xminDiam,xminNb];

% Algorithmic parameters
alpha=1;
c1=2; % cognitive component
c2=2; % social component
dt=1;
inertiaWeight=1.3;
beta=0.99;

maxVelocity=(xmax-xmin)/dt;
dvs=10;

%% Generate position and velocity vector of each particle
numberOfParticles=70;
numberOfDimensionSpace=14;    
nMaxIter=60;

PositionMatrix=zeros(numberOfParticles,numberOfDimensionSpace);
velocityMatrix=zeros(numberOfParticles,numberOfDimensionSpace);
for i=1:numberOfParticles
    for j=1:numberOfDimensionSpace
        r=rand;
        PositionMatrix(i,j)=xmin(j)+fix(r*(xmax(j)-(xmin(j)-1)));
        velocityMatrix(i,j)=alpha/dt*(-(xmax(j)-xmin(j))*0.5+r*(xmax(j)-xmin(j)));
    end
end

cbestLeft=[];

iteration=0;
bestPerformance=inf;
bestPosition=zeros(1,numberOfDimensionSpace);

performance=zeros(numberOfParticles,1);
bestPositionSwarmMatrix=PositionMatrix;
bestPerformanceSwarm=inf(numberOfParticles,1);
for j=1:nMaxIter
    iteration=iteration+1;

    % Determine the best position and best performance

    for i=1:numberOfParticles
        position=PositionMatrix(i,:); % diameter combo
        
        db1l=rebarAvailable(position(1),2);
        db2l=rebarAvailable(position(2),2);
        db3l=rebarAvailable(position(3),2);
        db4l=rebarAvailable(position(4),2);
        db5l=rebarAvailable(position(5),2);
        db6l=rebarAvailable(position(6),2);
        db7l=rebarAvailable(position(7),2);
        db8l=rebarAvailable(position(8),2);
        
        dbc=[db1l,db2l,db3l,db4l,db5l,db6l,db7l,db8l];
        abc=dbc.^2*pi/4;
        bpl=b-2*brec-2*dvs-max(dbc(1:6));
        
        nb2l=position(9);
        nb3l=position(10);
        nb4l=position(11);
        nb5l=position(12);
        nb6l=position(13);
        nb7l=2;
        nb8l=position(14);
        
        nbc=[2,nb2l,nb3l,nb4l,nb5l,nb6l,nb7l,nb8l];
        Abr=sum(abc.*nbc);
        performance(i)=Abr;
        
        %% Rebar separation
        % Left cross-section
        sepRebarl1=(bpl)/((2+nb2l)-1);
        if nb3l+nb4l>1
            sepRebarl2=(bpl)/(nb3l+nb4l-1);
        else
            sepRebarl2=1e10;
        end
        if nb5l+nb6l>1
            sepRebarl3=(bpl)/(nb5l+nb6l-1);
        else
            sepRebarl3=1e10;
        end
        % Rebar separation in compression zone
        if nb7l+nb8l>1
            sepRebarl4=(bpl)/(nb7l+nb8l-1);
        else
            sepRebarl4=1e10;
        end
        sepRebar=[sepRebarl1,sepRebarl2,sepRebarl3,sepRebarl4];
        
        [sepMin1l,sepMax1l]=sepMinMaxHK13([db1l,db2l],hagg,0);
        [sepMin2l,sepMax2l]=sepMinMaxHK13([db3l,db4l],hagg,0);
        [sepMin3l,sepMax3l]=sepMinMaxHK13([db5l,db6l],hagg,0);
        [sepMin4l,sepMax4l]=sepMinMaxHK13([db7l,db8l],hagg,0);
        
        sepMin=[sepMin1l,sepMin2l,sepMin3l,sepMin4l];
        
        % Rebars in compression
        [distrRebarLeftComp,listRebarDiamLeftComp]=distrRebarRecBeam([nbc(7:8),zeros(1,4)],...
            [[db7l,db8l],zeros(1,4)],b,h,brec,hrec,25,sepRebar(1,4));
        
        % Rebars in tension
        [distrRebarLeftTen,listRebarDiamsLeftTen]=distrRebarRecBeam...
            (nbc(1:6),dbc(1:6),b,h,brec,hrec,25,sepRebar(1,1));
        
        distrRebarLeft=[distrRebarLeftTen;
                        -distrRebarLeftComp];
        listRebarDiamLeft=[listRebarDiamsLeftTen;
                          listRebarDiamLeftComp];
                      
        [Eff,MrLeft,cLeft]=EfRecBeamBars(Mul,fcu,Es,fy,h,b,distrRebarLeft,...
                                     listRebarDiamLeft,hrec);
        [ccl]=rebarDistrConstr3LayerRecBeam(bpl,nbc(1:6));
        
        if all([performance(i)<bestPerformance,ccl==1, Eff<1, ...
                Abr>=amin, Abr<=amax, sepMin(1)<=sepRebar(1,1), ...
                sepMin(4)<=sepRebar(1,4)])
            
            bestPerformance=performance(i);
            bestsepRebar=sepRebar;
            bestPosition=position;
            bestdbc8=dbc;
            bestEfL=Eff;
            bestMrLeft=MrLeft;
            cbestLeft=cLeft;
            bestListRebarDiamLeft=listRebarDiamLeft;
            bestDistrRebarLeft=distrRebarLeft;
            bestnbt=nbc;
            
            bestPerformanceSwarm(i)=performance(i);
            bestPositionSwarmMatrix(i,:)=position;
        end
    end
    performanceHist(j)=bestPerformance;
    
    %% Global best position
    globalBestPosition=bestPosition;
    globalBestPerformance=bestPerformance;
    
    %% Velocity and position uptdating 

    for i=1:numberOfParticles
        q=rand;
        r=rand;
        %% Velocity updating
        for j=1:numberOfDimensionSpace
            velocityMatrix(i,j)=inertiaWeight*velocityMatrix(i,j)+...
                c1*q*((bestPositionSwarmMatrix(i,j)-PositionMatrix(i,j))/dt)+...
                c2*r*((bestPosition(j)-PositionMatrix(i,j)));
            
            absolouteValueVelocity(j)=abs(velocityMatrix(i,j));
            valueVelocity(j)=velocityMatrix(i,j);
            
            if (absolouteValueVelocity(j)>maxVelocity(j))
                    velocityMatrix(i,j)=maxVelocity(j);
            end
            if (valueVelocity(j)<-maxVelocity(j))
                    velocityMatrix(i,j)=-maxVelocity(j);
            end
        end

        %% Position uptdating
        for j=1:numberOfDimensionSpace
            tj=fix(PositionMatrix(i,j)+velocityMatrix(i,j)*dt);
            if tj<xmin(j)
                tj=xmin(j);
            elseif tj>xmax(j)
                tj=xmax(j);
            end
            PositionMatrix(i,j)=tj;
        end
    end
    
    if(inertiaWeight*beta<0.3)
        inertiaWeight=inertiaWeight;
    else
        inertiaWeight=inertiaWeight*beta;
    end
end

% Optima Convergence Graph
if graphConvPlot==1 && isempty(cbestLeft)==0
    figure(4)
    if sum(nbc)==0
        plot(1:1:nMaxIter,...
           performanceHist.*wac,'-o')
        hold on
        legend('Span 1')
    else
        plot(1:1:nMaxIter,...
           performanceHist.*wac,'-o','DisplayName',strcat('Span',num2str(ispan)))
       
        hold on
    end
    xlabel('Iteration')
    ylabel('Rebar weight (N)')
    title({'Optimum convergence of reinforcement weight for a concrete beam';
            'PSO'})
    hold on
    
    if layPlt==1
        plotBeamRebar1Sec(h,b,bestDistrRebarLeft,bestListRebarDiamLeft);
    end
end
if isempty(cbestLeft)==1
    bestPerformance=performance(i); % rebar volume (mm^3)
    
    bestsepRebar=0;
    bestPosition=[];
    bestdbc8=zeros(1,8);
    bestEfL=0;
    bestMrLeft=0;
    cbestLeft=0;
    bestListRebarDiamLeft=[];
    bestDistrRebarLeft=[];
    bestnbt=zeros(1,8);
end