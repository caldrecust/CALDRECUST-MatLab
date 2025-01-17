function plotBeamReinforcedSection(h,b,rebarAvailable,dispositionRebar,...
                                barDiamTension,barDiamComp)

%------------------------------------------------------------------------
% Syntax:
% beamReinforcedSection(h,b,rebarAvailable,dispositionRebar,barDiamTension,...
%                       barDiamComp)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: To plot the reinforcement of a rectangular cross-section.
% 
% INPUT:  barDiamTension,barDiamComp:   Vectors that contain the list of 
%                                       diameter sizes of rebar, both for
%                                       tension and compression, 
%                                       respectively. The vectors size is 
%                                       of one column with nrebar rows 
%                                       containing the diameter size,
%                                       according to the available commercial
%                                       rebars
%
%         dispositionRebar:             local coordinates of rebars over
%                                       the cross-section
%
%         b,h:                          cross-section dimensions
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2025-02-05
%                School of Engineering
%                The Hong Kong University of Science and Technology (HKUST)
%------------------------------------------------------------------------

%% Cross-section borders
coordConerSection=[0.5*b 0.5*h;
                         0.5*b -0.5*h;
                         -0.5*b -0.5*h;
                         -0.5*b 0.5*h;
                         0.5*b 0.5*h];
                     
x=coordConerSection(:,1);
y=coordConerSection(:,2);

nbars1=length(barDiamTension);
nbars2=length(barDiamComp);

%% Rebar
t1=[];
t2=[];
t3=[];
t4=[];
t5=[];
t6=[];
t7=[];
t8=[];
t9=[];

dispVar1x=[];
dispVar1y=[];

dispVar2x=[];
dispVar2y=[];

dispVar3x=[];
dispVar3y=[];

dispVar4x=[];
dispVar4y=[];

dispVar5x=[];
dispVar5y=[];

dispVar6x=[];
dispVar6y=[];

dispVar7x=[];
dispVar7y=[];

dispVar8x=[];
dispVar8y=[];

dispVar9x=[];
dispVar9y=[];

nbars=length(dispositionRebar(:,1));
barTypes=[];
for i=1:nbars1
    barTypes=[barTypes,barDiamTension(i)];
end

for i=1:nbars2
    barTypes=[barTypes,barDiamComp(i)];
end

for j=1:nbars
    if barTypes(j)==1
        t1=[t1,1];
        dispVar1x=[dispVar1x,dispositionRebar(j,1)];
        dispVar1y=[dispVar1y,dispositionRebar(j,2)];
    elseif barTypes(j)==2
        t2=[t2,2];
        dispVar2x=[dispVar2x,dispositionRebar(j,1)];
        dispVar2y=[dispVar2y,dispositionRebar(j,2)];
    elseif barTypes(j)==3
        t3=[t3,3];
        dispVar3x=[dispVar3x,dispositionRebar(j,1)];
        dispVar3y=[dispVar3y,dispositionRebar(j,2)];
    elseif barTypes(j)==4
        t4=[t4,4];
        dispVar4x=[dispVar4x,dispositionRebar(j,1)];
        dispVar4y=[dispVar4y,dispositionRebar(j,2)];
    elseif barTypes(j)==5
        t5=[t5,5];
        dispVar5x=[dispVar5x,dispositionRebar(j,1)];
        dispVar5y=[dispVar5y,dispositionRebar(j,2)];
    elseif barTypes(j)==6
        t6=[t6,6];
        dispVar6x=[dispVar6x,dispositionRebar(j,1)];
        dispVar6y=[dispVar6y,dispositionRebar(j,2)];
    elseif barTypes(j)==7
        t7=[t7,7];
        dispVar7x=[dispVar7x,dispositionRebar(j,1)];
        dispVar7y=[dispVar7y,dispositionRebar(j,2)];
    elseif barTypes(j)==8
        t8=[t8,8];
        dispVar8x=[dispVar8x,dispositionRebar(j,1)];
        dispVar8y=[dispVar8y,dispositionRebar(j,2)];
    elseif barTypes(j)==9
        t9=[t9,9];
        dispVar9x=[dispVar9x,dispositionRebar(j,1)];
        dispVar9y=[dispVar9y,dispositionRebar(j,2)];
    end
end
        
figure(2)
plot(x,y,'k')
hold on
xlabel('Width')
ylabel('Height')
title('Rectangular Cross-Section')
legend('Section borders')
axis([-(b+20) b+20 -(h+5) h+5])

if isempty(t1)~=1
    figure(2)
    plot(dispVar1x,dispVar1y,'r o','linewidth',1,'MarkerFaceColor','red',...
        'DisplayName',strcat('Bar Diam: ',num2str(rebarAvailable(1,1))));
end
if isempty(t2)~=1
    figure(2)
    plot(dispVar2x,dispVar2y,'b o','linewidth',1,'MarkerFaceColor','blue',...
        'DisplayName',strcat('Bar Diam: ',num2str(rebarAvailable(2,1))));
end
if isempty(t3)~=1
    figure(2)
    plot(dispVar3x,dispVar3y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.05 0.205 0.05]','DisplayName',strcat('Bar Diam: ',num2str(rebarAvailable(3,1))));
end
if isempty(t4)~=1
    figure(2)
    plot(dispVar4x,dispVar4y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.072 0.061 0.139]','DisplayName',strcat('Bar Diam: ',num2str(rebarAvailable(4,1))));
end
if isempty(t5)~=1
    figure(2)
    plot(dispVar5x,dispVar5y,'k o','linewidth',1,'MarkerFaceColor',...
        'black','DisplayName',strcat('Bar Diam: ',num2str(rebarAvailable(5,1))));
end
if isempty(t6)~=1
    figure(2)
    plot(dispVar6x,dispVar6y,'m o','linewidth',1,'MarkerFaceColor',...
        'magenta','DisplayName',strcat('Bar Diam: ',num2str(rebarAvailable(6,1))));
end
if isempty(t7)~=1
    figure(2)
    plot(dispVar7x,dispVar7y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.255 0.069 0]','DisplayName',strcat('Bar Diam: ',num2str(rebarAvailable(7,1))));
end
if isempty(t8)~=1
    figure(2)
    plot(dispVar8x,dispVar8y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.255 0.069 0]','DisplayName',strcat('Bar Diam: ',num2str(rebarAvailable(8,1))));
end
if isempty(t9)~=1
    figure(2)
    plot(dispVar9x,dispVar9y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.255 0.069 0]','DisplayName',strcat('Bar Diam: ',num2str(rebarAvailable(9,1))));
end