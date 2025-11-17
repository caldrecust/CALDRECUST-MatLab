function [bestPosition,bestPerformance,restric]=ISROptmRecBeam3LayersPSO...
                            (b,h,Cc,fcu,Es,fy,Mu,nparticles,niter,pltconv)


%% Restrictions

% Max and min reinforcement
Amin=0.003*b*h;
Amax=0.025*b*h;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Particle Swarm Optimization%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Search space
dvs=10;
bp=b-2*Cc-2*dvs;
tmin=Amin/bp;
tmax=Amax/bp;
xMaxMin=[tmin,tmax;
         0,tmax;
         0,tmax];

alpha=1;
c1=2; % cognitive component
c2=2; % social component
dt=1;
inertiaWeight=1.3;
beta=0.99;

maxVelocity=(xMaxMin(:,2)-xMaxMin(:,1))/dt;

numberOfDimensionSpace=3;

%% Optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Generate position and velocity vector of each particle%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
PositionMatrix=zeros(nparticles,numberOfDimensionSpace);
velocityMatrix=zeros(nparticles,numberOfDimensionSpace);
for i=1:nparticles
    for j=1:numberOfDimensionSpace
        r=rand;
        PositionMatrix(i,j)=xMaxMin(j,1)+r*(xMaxMin(j,2)-xMaxMin(j,1));
        velocityMatrix(i,j)=alpha/dt*(-(xMaxMin(j,2)-xMaxMin(j,1))*0.5+r*(xMaxMin(j,2)-xMaxMin(j,1)));
    end
end

%%%%%%%%%%%
%Main Loop%
%%%%%%%%%%%

restric=0;
bestPerformance=inf;
bestPosition=zeros(1,numberOfDimensionSpace);
position=zeros(1,numberOfDimensionSpace);
performance=zeros(nparticles,1);
bestPositionSwarmMatrix=PositionMatrix;
bestPerformanceSwarm=inf(nparticles,1);
for iter=1:niter

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Determine the best position and best performance%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for i=1:nparticles
        position=PositionMatrix(i,:); % t widths
        
        % Cross-section area of rebar
        
        performance(i)=AreaISRecBeam3Layers(position,b,Cc);
        [tdistr]=distrRecBeam3tLayers(b,h,Cc,position,25);
        [Eff,Mrt,c]=EfRecBeamt(Mu,fcu,Es,fy,h,b,tdistr,position,Cc);
        
        % here is where all the feasible solutions get in
        if (performance(i)<bestPerformance && ...
                all([performance(i)>=Amin,performance(i)<=Amax,Eff<1]))
            
            bestPerformance=performance(i);
            bestPosition=PositionMatrix(i,:);
            restric=1;
        end
        listPerformance(iter,1)=bestPerformance;
        if (performance(i)<bestPerformanceSwarm(i) && ...
                all([performance(i)>=Amin,performance(i)<=Amax,Eff<1]))
            bestPerformanceSwarm(i)=performance(i);
            bestPositionSwarmMatrix(i,:)=position;
        end
    end
    
    % global best position 
    globalBestPosition=bestPosition;
    globalBestPerformance=bestPerformance;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % velocity and position uptdating %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for i=1:nparticles
        q=rand;
        r=rand;
        % velocity updating
        for j=1:numberOfDimensionSpace
            velocityMatrix(i,j)=inertiaWeight*velocityMatrix(i,j)+...
                c1*q*((bestPositionSwarmMatrix(i,j)-PositionMatrix(i,j))/dt)+...
                c2*r*((bestPosition(j)-PositionMatrix(i,j)));
        
            absValVel(j,1)=abs(velocityMatrix(i,j));
            valVel(j,1)=velocityMatrix(i,j);
            if (absValVel(j,1)>maxVelocity(j,1))
                    velocityMatrix(i,j)=maxVelocity(j,1);
            end
            if (valVel(j,1)<-maxVelocity(j,1))
                    velocityMatrix(i,1)=-maxVelocity(j,1);
            end
        end
        %%%%%%%%%%%%%%%%%%%%
        %position uptdating%
        %%%%%%%%%%%%%%%%%%%%

        for j=1:numberOfDimensionSpace
            PositionMatrix(i,j)=PositionMatrix(i,j)+velocityMatrix(i,j)*dt;
            if PositionMatrix(i,j)<xMaxMin(j,1)
                PositionMatrix(i,j)=xMaxMin(j,1);
            end
            if  PositionMatrix(i,j)>xMaxMin(j,2)
                PositionMatrix(i,j)=xMaxMin(j,2);
            end 
            
        end
    end
    
    if(inertiaWeight*beta<0.3)
        inertiaWeight=inertiaWeight;
    else
        inertiaWeight=inertiaWeight*beta;
    end

end

if restric==0
    bestPerformance=0;
end
if restric~=0 && pltconv==1
    figure(1)
    plot(1:1:niter,listPerformance,'k o-')
    title('ISR optimization - PSO')
    hold on
end