function plotBeamRebar1Sec(h,b,distrRebar,barDiamList)

%------------------------------------------------------------------------
% Syntax:
% plotBeamRebar1Sec(h,b,dispositionRebar,barDiamList)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: To plot the reinforcement of a rectangular cross-section.
% 
% INPUT:  barDiamList:                  Vectors that contain the list of 
%                                       diameter sizes of rebar, both for
%                                       tension and compression, 
%                                       respectively. The vectors size is 
%                                       of one column with nrebar rows 
%                                       containing the diameter size,
%                                       according to the available commercial
%                                       rebars
%
%         distrRebar:                   local coordinates of rebars over
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
t10=[];

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

dispVar10x=[];
dispVar10y=[];

nbars=length(distrRebar(:,1));

for j=1:nbars
    if barDiamList(j)==6
        t1=[t1,1];
        dispVar1x=[dispVar1x,distrRebar(j,1)];
        dispVar1y=[dispVar1y,distrRebar(j,2)];
    elseif barDiamList(j)==8
        t2=[t2,2];
        dispVar2x=[dispVar2x,distrRebar(j,1)];
        dispVar2y=[dispVar2y,distrRebar(j,2)];
    elseif barDiamList(j)==10
        t3=[t3,3];
        dispVar3x=[dispVar3x,distrRebar(j,1)];
        dispVar3y=[dispVar3y,distrRebar(j,2)];
    elseif barDiamList(j)==12
        t4=[t4,4];
        dispVar4x=[dispVar4x,distrRebar(j,1)];
        dispVar4y=[dispVar4y,distrRebar(j,2)];
    elseif barDiamList(j)==16
        t5=[t5,5];
        dispVar5x=[dispVar5x,distrRebar(j,1)];
        dispVar5y=[dispVar5y,distrRebar(j,2)];
    elseif barDiamList(j)==20
        t6=[t6,6];
        dispVar6x=[dispVar6x,distrRebar(j,1)];
        dispVar6y=[dispVar6y,distrRebar(j,2)];
    elseif barDiamList(j)==25
        t7=[t7,7];
        dispVar7x=[dispVar7x,distrRebar(j,1)];
        dispVar7y=[dispVar7y,distrRebar(j,2)];
    elseif barDiamList(j)==32
        t8=[t8,8];
        dispVar8x=[dispVar8x,distrRebar(j,1)];
        dispVar8y=[dispVar8y,distrRebar(j,2)];
    elseif barDiamList(j)==40
        t9=[t9,9];
        dispVar9x=[dispVar9x,distrRebar(j,1)];
        dispVar9y=[dispVar9y,distrRebar(j,2)];
    elseif barDiamList(j)==50
        t10=[t10,10];
        dispVar10x=[dispVar10x,distrRebar(j,1)];
        dispVar10y=[dispVar10y,distrRebar(j,2)];
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
    plot(dispVar1x,dispVar1y,'o','MarkerEdgeColor','[1 0 0]','MarkerFaceColor','[1 0 0]',...
        'DisplayName',strcat('Bar Diam: ',num2str(6)));
end
if isempty(t2)~=1
    figure(2)
    plot(dispVar2x,dispVar2y,'o','MarkerEdgeColor','[0 0 1]','MarkerFaceColor','[0 0 1]',...
        'DisplayName',strcat('Bar Diam: ',num2str(8)));
end
if isempty(t3)~=1
    figure(2)
    plot(dispVar3x,dispVar3y,'o','MarkerEdgeColor','[0.4940 0.1840 0.5560]','MarkerFaceColor',...
        '[0.4940 0.1840 0.5560]','DisplayName',strcat('Bar Diam: ',num2str(10)));
end
if isempty(t4)~=1
    figure(2)
    plot(dispVar4x,dispVar4y,'o','MarkerEdgeColor','[0 0.7 0.7]','MarkerFaceColor',...
        '[0 0.7 0.7]','DisplayName',strcat('Bar Diam: ',num2str(12)));
end
if isempty(t5)~=1
    figure(2)
    plot(dispVar5x,dispVar5y,'o','MarkerEdgeColor','black','MarkerFaceColor',...
        'black','DisplayName',strcat('Bar Diam: ',num2str(16)));
end
if isempty(t6)~=1
    figure(2)
    plot(dispVar6x,dispVar6y,'o','MarkerEdgeColor','[1 0 1]','MarkerFaceColor',...
        '[1 0 1]','DisplayName',strcat('Bar Diam: ',num2str(20)));
end
if isempty(t7)~=1
    figure(2)
    plot(dispVar7x,dispVar7y,'o','MarkerEdgeColor','[0.6350 0.0780 0.1840]','MarkerFaceColor',...
        '[0.6350 0.0780 0.1840]','DisplayName',strcat('Bar Diam: ',num2str(25)));
end
if isempty(t8)~=1
    figure(2)
    plot(dispVar8x,dispVar8y,'o','MarkerEdgeColor','[0.8500 0.3250 0.0980]','MarkerFaceColor',...
        '[0.8500 0.3250 0.0980]','DisplayName',strcat('Bar Diam: ',num2str(32)));
end
if isempty(t9)~=1
    figure(2)
    plot(dispVar9x,dispVar9y,'o','MarkerEdgeColor','[0.4660 0.6740 0.1880]','MarkerFaceColor',...
        '[0.4660 0.6740 0.1880]','DisplayName',strcat('Bar Diam: ',num2str(40)));
end
if isempty(t10)~=1
    figure(2)
    plot(dispVar9x,dispVar9y,'o','MarkerEdgeColor','[0.7 0 1]','MarkerFaceColor',...
        '[0.7 0 1]','DisplayName',strcat('Bar Diam: ',num2str(50)));
end