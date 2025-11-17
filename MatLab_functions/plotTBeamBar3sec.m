function plotTBeamBar3sec(bw,h,b,hf,distRebarLeft,ListRebarDiamLeft,...
         distRebarMid,ListRebarDiamMid,distRebarRight,ListRebarDiamRight,nfig)

CoordCorners=[-0.5*b 0.5*h;
             0.5*b 0.5*h;
             0.5*b 0.5*h-hf;
             0.5*bw 0.5*h-hf;
             0.5*bw -0.5*h;
             -0.5*bw -0.5*h;
             -0.5*bw 0.5*h-hf;
             -0.5*b 0.5*h-hf;
             -0.5*b 0.5*h];

x=CoordCorners(:,1);
y=CoordCorners(:,2);

%------------------------------ beam plot -------------------------%

%% Left cross-section
nbars=length(ListRebarDiamLeft);

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
for j=1:nbars
    if ListRebarDiamLeft(j)==6
        t1=[t1,1];
        dispVar1x=[dispVar1x,distRebarLeft(j,1)];
        dispVar1y=[dispVar1y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==8
        t2=[t2,2];
        dispVar2x=[dispVar2x,distRebarLeft(j,1)];
        dispVar2y=[dispVar2y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==10
        t3=[t3,3];
        dispVar3x=[dispVar3x,distRebarLeft(j,1)];
        dispVar3y=[dispVar3y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==12
        t4=[t4,4];
        dispVar4x=[dispVar4x,distRebarLeft(j,1)];
        dispVar4y=[dispVar4y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==16
        t5=[t5,5];
        dispVar5x=[dispVar5x,distRebarLeft(j,1)];
        dispVar5y=[dispVar5y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==20
        t6=[t6,6];
        dispVar6x=[dispVar6x,distRebarLeft(j,1)];
        dispVar6y=[dispVar6y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==25
        t7=[t7,7];
        dispVar7x=[dispVar7x,distRebarLeft(j,1)];
        dispVar7y=[dispVar7y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==32
        t8=[t8,8];
        dispVar8x=[dispVar8x,distRebarLeft(j,1)];
        dispVar8y=[dispVar8y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==40
        t9=[t9,9];
        dispVar9x=[dispVar9x,distRebarLeft(j,1)];
        dispVar9y=[dispVar9y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==50
        t10=[t10,10];
        dispVar10x=[dispVar10x,distRebarLeft(j,1)];
        dispVar10y=[dispVar10y,distRebarLeft(j,2)];
    end
end
figure(nfig)
subplot(1,3,1)
plot(x,y,'k -','linewidth',1)
hold on
xlabel('x´')
ylabel('y´')
title({'Rectangular reinforced beam';'Left section'})
legend('Beam´s boundaries')
axis([-(b+10) b+10 -(h+10) h+10])
hold on

if isempty(t1)~=1
    subplot(1,3,1)
    figure(nfig)
    plot(dispVar1x,dispVar1y,'o','MarkerEdgeColor','[1 0 0]','MarkerFaceColor','[1 0 0]',...
        'DisplayName',strcat('Bar Diam: ',num2str(6)));
end
if isempty(t2)~=1
    subplot(1,3,1)
    figure(nfig)
    plot(dispVar2x,dispVar2y,'o','MarkerEdgeColor','[0 0 1]','MarkerFaceColor','[0 0 1]',...
        'DisplayName',strcat('Bar Diam: ',num2str(8)));
end
if isempty(t3)~=1
    subplot(1,3,1)
    figure(nfig)
    plot(dispVar3x,dispVar3y,'o','MarkerEdgeColor','[0.4940 0.1840 0.5560]','MarkerFaceColor',...
        '[0.4940 0.1840 0.5560]','DisplayName',strcat('Bar Diam: ',num2str(10)));
end
if isempty(t4)~=1
    subplot(1,3,1)
    figure(nfig)
    plot(dispVar4x,dispVar4y,'o','MarkerEdgeColor','[0 0.7 0.7]','MarkerFaceColor',...
        '[0 0.7 0.7]','DisplayName',strcat('Bar Diam: ',num2str(12)));
end
if isempty(t5)~=1
    subplot(1,3,1)
    figure(nfig)
    plot(dispVar5x,dispVar5y,'o','MarkerEdgeColor','black','MarkerFaceColor',...
        'black','DisplayName',strcat('Bar Diam: ',num2str(16)));
end
if isempty(t6)~=1
    subplot(1,3,1)
    figure(nfig)
    plot(dispVar6x,dispVar6y,'o','MarkerEdgeColor','[1 0 1]','MarkerFaceColor',...
        '[1 0 1]','DisplayName',strcat('Bar Diam: ',num2str(20)));
end
if isempty(t7)~=1
    subplot(1,3,1)
    figure(nfig)
    plot(dispVar7x,dispVar7y,'o','MarkerEdgeColor','[0.6350 0.0780 0.1840]','MarkerFaceColor',...
        '[0.6350 0.0780 0.1840]','DisplayName',strcat('Bar Diam: ',num2str(25)));
end
if isempty(t8)~=1
    subplot(1,3,1)
    figure(nfig)
    plot(dispVar8x,dispVar8y,'o','MarkerEdgeColor','[0.8500 0.3250 0.0980]','MarkerFaceColor',...
        '[0.8500 0.3250 0.0980]','DisplayName',strcat('Bar Diam: ',num2str(32)));
end
if isempty(t9)~=1
    subplot(1,3,1)
    figure(nfig)
    plot(dispVar9x,dispVar9y,'o','MarkerEdgeColor','[0.4660 0.6740 0.1880]','MarkerFaceColor',...
        '[0.4660 0.6740 0.1880]','DisplayName',strcat('Bar Diam: ',num2str(40)));
end

if isempty(t10)~=1
    subplot(1,3,1)
    figure(nfig)
    plot(dispVar9x,dispVar9y,'o','MarkerEdgeColor','[0.7 0 1]','MarkerFaceColor',...
        '[0.7 0 1]','DisplayName',strcat('Bar Diam: ',num2str(50)));
end
%% Mid cross-section
nbars=length(ListRebarDiamMid);

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
for j=1:nbars
    if ListRebarDiamMid(j)==6
        t1=[t1,1];
        dispVar1x=[dispVar1x,distRebarMid(j,1)];
        dispVar1y=[dispVar1y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==8
        t2=[t2,2];
        dispVar2x=[dispVar2x,distRebarMid(j,1)];
        dispVar2y=[dispVar2y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==10
        t3=[t3,3];
        dispVar3x=[dispVar3x,distRebarMid(j,1)];
        dispVar3y=[dispVar3y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==12
        t4=[t4,4];
        dispVar4x=[dispVar4x,distRebarMid(j,1)];
        dispVar4y=[dispVar4y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==16
        t5=[t5,5];
        dispVar5x=[dispVar5x,distRebarMid(j,1)];
        dispVar5y=[dispVar5y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==20
        t6=[t6,6];
        dispVar6x=[dispVar6x,distRebarMid(j,1)];
        dispVar6y=[dispVar6y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==25
        t7=[t7,7];
        dispVar7x=[dispVar7x,distRebarMid(j,1)];
        dispVar7y=[dispVar7y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==32
        t8=[t8,8];
        dispVar8x=[dispVar8x,distRebarMid(j,1)];
        dispVar8y=[dispVar8y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==40
        t9=[t9,9];
        dispVar9x=[dispVar9x,distRebarMid(j,1)];
        dispVar9y=[dispVar9y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==50
        t10=[t10,10];
        dispVar10x=[dispVar10x,distRebarMid(j,1)];
        dispVar10y=[dispVar10y,distRebarMid(j,2)];
    end
end
figure(nfig)
subplot(1,3,2)
plot(x,y,'k -','linewidth',1)
hold on
xlabel('x´')
ylabel('y´')
title({'Rectangular reinforced beam';'Mid section'})
legend('Beam´s boundaries')
axis([-(b+10) b+10 -(h+10) h+10])
hold on

if isempty(t1)~=1
    subplot(1,3,2)
    figure(nfig)
    plot(dispVar1x,dispVar1y,'o','MarkerEdgeColor','[1 0 0]','MarkerFaceColor','[1 0 0]',...
        'DisplayName',strcat('Bar Diam: ',num2str(6)));
end
if isempty(t2)~=1
    subplot(1,3,2)
    figure(nfig)
    plot(dispVar2x,dispVar2y,'o','MarkerEdgeColor','[0 0 1]','MarkerFaceColor','[0 0 1]',...
        'DisplayName',strcat('Bar Diam: ',num2str(8)));
end
if isempty(t3)~=1
    subplot(1,3,2)
    figure(nfig)
    plot(dispVar3x,dispVar3y,'o','MarkerEdgeColor','[0.4940 0.1840 0.5560]','MarkerFaceColor',...
        '[0.4940 0.1840 0.5560]','DisplayName',strcat('Bar Diam: ',num2str(10)));
end
if isempty(t4)~=1
    subplot(1,3,2)
    figure(nfig)
    plot(dispVar4x,dispVar4y,'o','MarkerEdgeColor','[0 0.7 0.7]','MarkerFaceColor',...
        '[0 0.7 0.7]','DisplayName',strcat('Bar Diam: ',num2str(12)));
end
if isempty(t5)~=1
    subplot(1,3,2)
    figure(nfig)
    plot(dispVar5x,dispVar5y,'o','MarkerEdgeColor','black','MarkerFaceColor',...
        'black','DisplayName',strcat('Bar Diam: ',num2str(16)));
end
if isempty(t6)~=1
    subplot(1,3,2)
    figure(nfig)
    plot(dispVar6x,dispVar6y,'o','MarkerEdgeColor','[1 0 1]','MarkerFaceColor',...
        '[1 0 1]','DisplayName',strcat('Bar Diam: ',num2str(20)));
end
if isempty(t7)~=1
    subplot(1,3,2)
    figure(nfig)
    plot(dispVar7x,dispVar7y,'o','MarkerEdgeColor','[0.6350 0.0780 0.1840]','MarkerFaceColor',...
        '[0.6350 0.0780 0.1840]','DisplayName',strcat('Bar Diam: ',num2str(25)));
end
if isempty(t8)~=1
    subplot(1,3,2)
    figure(nfig)
    plot(dispVar8x,dispVar8y,'o','MarkerEdgeColor','[0.8500 0.3250 0.0980]','MarkerFaceColor',...
        '[0.8500 0.3250 0.0980]','DisplayName',strcat('Bar Diam: ',num2str(32)));
end
if isempty(t9)~=1
    subplot(1,3,2)
    figure(nfig)
    plot(dispVar9x,dispVar9y,'o','MarkerEdgeColor','[0.4660 0.6740 0.1880]','MarkerFaceColor',...
        '[0.4660 0.6740 0.1880]','DisplayName',strcat('Bar Diam: ',num2str(40)));
end

if isempty(t10)~=1
    subplot(1,3,2)
    figure(nfig)
    plot(dispVar9x,dispVar9y,'o','MarkerEdgeColor','[0.7 0 1]','MarkerFaceColor',...
        '[0.7 0 1]','DisplayName',strcat('Bar Diam: ',num2str(50)));
end
%% Right cross-section
nbars=length(ListRebarDiamRight);

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

for j=1:nbars
    if ListRebarDiamRight(j)==6
        t1=[t1,1];
        dispVar1x=[dispVar1x,distRebarRight(j,1)];
        dispVar1y=[dispVar1y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==8
        t2=[t2,2];
        dispVar2x=[dispVar2x,distRebarRight(j,1)];
        dispVar2y=[dispVar2y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==10
        t3=[t3,3];
        dispVar3x=[dispVar3x,distRebarRight(j,1)];
        dispVar3y=[dispVar3y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==12
        t4=[t4,4];
        dispVar4x=[dispVar4x,distRebarRight(j,1)];
        dispVar4y=[dispVar4y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==16
        t5=[t5,5];
        dispVar5x=[dispVar5x,distRebarRight(j,1)];
        dispVar5y=[dispVar5y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==20
        t6=[t6,6];
        dispVar6x=[dispVar6x,distRebarRight(j,1)];
        dispVar6y=[dispVar6y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==25
        t7=[t7,7];
        dispVar7x=[dispVar7x,distRebarRight(j,1)];
        dispVar7y=[dispVar7y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==32
        t8=[t8,8];
        dispVar8x=[dispVar8x,distRebarRight(j,1)];
        dispVar8y=[dispVar8y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==40
        t9=[t9,9];
        dispVar9x=[dispVar9x,distRebarRight(j,1)];
        dispVar9y=[dispVar9y,distRebarRight(j,2)];
    
    elseif ListRebarDiamRight(j)==50
        t10=[t10,10];
        dispVar10x=[dispVar10x,distRebarRight(j,1)];
        dispVar10y=[dispVar10y,distRebarRight(j,2)];
    end
end
figure(nfig)
subplot(1,3,3)
plot(x,y,'k -','linewidth',1)
hold on
xlabel('x´')
ylabel('y´')
title({'Rectangular reinforced beam';'Right section'})
legend('Beam´s boundaries')
axis([-(b+10) b+10 -(h+10) h+10])
hold on

if isempty(t1)~=1
    subplot(1,3,3)
    figure(nfig)
    plot(dispVar1x,dispVar1y,'o','MarkerEdgeColor','[1 0 0]','MarkerFaceColor','[1 0 0]',...
        'DisplayName',strcat('Bar Diam: ',num2str(6)));
end
if isempty(t2)~=1
    subplot(1,3,3)
    figure(nfig)
    plot(dispVar2x,dispVar2y,'o','MarkerEdgeColor','[0 0 1]','MarkerFaceColor','[0 0 1]',...
        'DisplayName',strcat('Bar Diam: ',num2str(8)));
end
if isempty(t3)~=1
    subplot(1,3,3)
    figure(nfig)
    plot(dispVar3x,dispVar3y,'o','MarkerEdgeColor','[0.4940 0.1840 0.5560]','MarkerFaceColor',...
        '[0.4940 0.1840 0.5560]','DisplayName',strcat('Bar Diam: ',num2str(10)));
end
if isempty(t4)~=1
    subplot(1,3,3)
    figure(nfig)
    plot(dispVar4x,dispVar4y,'o','MarkerEdgeColor','[0 0.7 0.7]','MarkerFaceColor',...
        '[0 0.7 0.7]','DisplayName',strcat('Bar Diam: ',num2str(12)));
end
if isempty(t5)~=1
    subplot(1,3,3)
    figure(nfig)
    plot(dispVar5x,dispVar5y,'o','MarkerEdgeColor','black','MarkerFaceColor',...
        'black','DisplayName',strcat('Bar Diam: ',num2str(16)));
end
if isempty(t6)~=1
    subplot(1,3,3)
    figure(nfig)
    plot(dispVar6x,dispVar6y,'o','MarkerEdgeColor','[1 0 1]','MarkerFaceColor',...
        '[1 0 1]','DisplayName',strcat('Bar Diam: ',num2str(20)));
end
if isempty(t7)~=1
    subplot(1,3,3)
    figure(nfig)
    plot(dispVar7x,dispVar7y,'o','MarkerEdgeColor','[0.6350 0.0780 0.1840]','MarkerFaceColor',...
        '[0.6350 0.0780 0.1840]','DisplayName',strcat('Bar Diam: ',num2str(25)));
end
if isempty(t8)~=1
    subplot(1,3,3)
    figure(nfig)
    plot(dispVar8x,dispVar8y,'o','MarkerEdgeColor','[0.8500 0.3250 0.0980]','MarkerFaceColor',...
        '[0.8500 0.3250 0.0980]','DisplayName',strcat('Bar Diam: ',num2str(32)));
end
if isempty(t9)~=1
    subplot(1,3,3)
    figure(nfig)
    plot(dispVar9x,dispVar9y,'o','MarkerEdgeColor','[0.4660 0.6740 0.1880]','MarkerFaceColor',...
        '[0.4660 0.6740 0.1880]','DisplayName',strcat('Bar Diam: ',num2str(40)));
end
if isempty(t10)~=1
    subplot(1,3,3)
    figure(nfig)
    plot(dispVar10x,dispVar10y,'o','MarkerEdgeColor','[0.7 0 1]','MarkerFaceColor',...
        '[0.7 0 1]','DisplayName',strcat('Bar Diam: ',num2str(50)));
end