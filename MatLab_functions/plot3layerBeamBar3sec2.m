function plot3layerBeamBar3sec2(b,h,distRebarLeft,ListRebarDiamLeft,...
         distRebarMid,ListRebarDiamMid,distRebarRight,ListRebarDiamRight)

CoordCorners=[-0.5*b, 0.5*h;
               0.5*b, 0.5*h;
               0.5*b, -0.5*h;
              -0.5*b, -0.5*h;
              -0.5*b, 0.5*h];

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

for j=1:nbars
    if ListRebarDiamLeft(j)==4/8.*2.54
        t1=[t1,1];
        dispVar1x=[dispVar1x,distRebarLeft(j,1)];
        dispVar1y=[dispVar1y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==5/8.*2.54
        t2=[t2,2];
        dispVar2x=[dispVar2x,distRebarLeft(j,1)];
        dispVar2y=[dispVar2y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==6/8.*2.54
        t3=[t3,3];
        dispVar3x=[dispVar3x,distRebarLeft(j,1)];
        dispVar3y=[dispVar3y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==8/8.*2.54
        t4=[t4,4];
        dispVar4x=[dispVar4x,distRebarLeft(j,1)];
        dispVar4y=[dispVar4y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==9/8.*2.54
        t5=[t5,5];
        dispVar5x=[dispVar5x,distRebarLeft(j,1)];
        dispVar5y=[dispVar5y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==10/8.*2.54
        t6=[t6,6];
        dispVar6x=[dispVar6x,distRebarLeft(j,1)];
        dispVar6y=[dispVar6y,distRebarLeft(j,2)];
    elseif ListRebarDiamLeft(j)==12/8.*2.54
        t7=[t7,7];
        dispVar7x=[dispVar7x,distRebarLeft(j,1)];
        dispVar7y=[dispVar7y,distRebarLeft(j,2)];
    end
end
figure(6)
subplot(1,3,1)
plot(x,y,'k -','linewidth',1)
hold on
xlabel('x´')
ylabel('y´')
title('Rectangular reinforced beam (Left section)')
legend('Beam´s boundaries')
axis([-(b+10) b+10 -(h+10) h+10])
hold on

if isempty(t1)~=1
    figure(6)
    subplot(1,3,1)
    plot(dispVar1x,dispVar1y,'r o','linewidth',1,'MarkerFaceColor','red',...
        'DisplayName',strcat('Bar Diam',num2str(4/8.*2.54)));
end
if isempty(t2)~=1
    figure(6)
    subplot(1,3,1)
    plot(dispVar2x,dispVar2y,'b o','linewidth',1,'MarkerFaceColor','blue',...
        'DisplayName',strcat('Bar Diam',num2str(5/8.*2.54)));
end
if isempty(t3)~=1
    figure(6)
    subplot(1,3,1)
    plot(dispVar3x,dispVar3y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.05 0.205 0.05]','DisplayName',strcat('Bar Diam',num2str(6/8.*2.54)));
end
if isempty(t4)~=1
    figure(6)
    subplot(1,3,1)
    plot(dispVar4x,dispVar4y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.072 0.061 0.139]','DisplayName',strcat('Bar Diam',num2str(8/8.*2.54)));
end
if isempty(t5)~=1
    figure(6)
    subplot(1,3,1)
    plot(dispVar5x,dispVar5y,'k o','linewidth',1,'MarkerFaceColor',...
        'black','DisplayName',strcat('Bar Diam',num2str(9/8.*2.54)));
end
if isempty(t6)~=1
    figure(6)
    subplot(1,3,1)
    plot(dispVar6x,dispVar6y,'m o','linewidth',1,'MarkerFaceColor',...
        'magenta','DisplayName',strcat('Bar Diam',num2str(10/8.*2.54)));
end
if isempty(t7)~=1
    figure(6)
    subplot(1,3,1)
    plot(dispVar7x,dispVar7y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.255 0.069 0]','DisplayName',strcat('Bar Diam',num2str(12/8.*2.54)));
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

for j=1:nbars
    if ListRebarDiamMid(j)==4/8.*2.54
        t1=[t1,1];
        dispVar1x=[dispVar1x,distRebarMid(j,1)];
        dispVar1y=[dispVar1y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==5/8.*2.54
        t2=[t2,2];
        dispVar2x=[dispVar2x,distRebarMid(j,1)];
        dispVar2y=[dispVar2y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==6/8.*2.54
        t3=[t3,3];
        dispVar3x=[dispVar3x,distRebarMid(j,1)];
        dispVar3y=[dispVar3y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==8/8.*2.54
        t4=[t4,4];
        dispVar4x=[dispVar4x,distRebarMid(j,1)];
        dispVar4y=[dispVar4y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==9/8.*2.54
        t5=[t5,5];
        dispVar5x=[dispVar5x,distRebarMid(j,1)];
        dispVar5y=[dispVar5y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==10/8.*2.54
        t6=[t6,6];
        dispVar6x=[dispVar6x,distRebarMid(j,1)];
        dispVar6y=[dispVar6y,distRebarMid(j,2)];
    elseif ListRebarDiamMid(j)==12/8.*2.54
        t7=[t7,7];
        dispVar7x=[dispVar7x,distRebarMid(j,1)];
        dispVar7y=[dispVar7y,distRebarMid(j,2)];
    end
end
figure(6)
subplot(1,3,2)
plot(x,y,'k -','linewidth',1)
hold on
xlabel('x´')
ylabel('y´')
title('Rectangular reinforced beam (Mid section)')
legend('Beam´s boundaries')
axis([-(b+10) b+10 -(h+10) h+10])
hold on

if isempty(t1)~=1
    figure(6)
    subplot(1,3,2)
    plot(dispVar1x,dispVar1y,'r o','linewidth',1,'MarkerFaceColor','red',...
        'DisplayName',strcat('Bar Diam',num2str(4/8.*2.54)));
end
if isempty(t2)~=1
    figure(6)
    subplot(1,3,2)
    plot(dispVar2x,dispVar2y,'b o','linewidth',1,'MarkerFaceColor','blue',...
        'DisplayName',strcat('Bar Diam',num2str(5/8.*2.54)));
end
if isempty(t3)~=1
    figure(6)
    subplot(1,3,2)
    plot(dispVar3x,dispVar3y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.05 0.205 0.05]','DisplayName',strcat('Bar Diam',num2str(6/8.*2.54)));
end
if isempty(t4)~=1
    figure(6)
    subplot(1,3,2)
    plot(dispVar4x,dispVar4y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.072 0.061 0.139]','DisplayName',strcat('Bar Diam',num2str(8/8.*2.54)));
end
if isempty(t5)~=1
    figure(6)
    subplot(1,3,2)
    plot(dispVar5x,dispVar5y,'k o','linewidth',1,'MarkerFaceColor',...
        'black','DisplayName',strcat('Bar Diam',num2str(9/8.*2.54)));
end
if isempty(t6)~=1
    figure(6)
    subplot(1,3,2)
    plot(dispVar6x,dispVar6y,'m o','linewidth',1,'MarkerFaceColor',...
        'magenta','DisplayName',strcat('Bar Diam',num2str(10/8.*2.54)));
end
if isempty(t7)~=1
    figure(6)
    subplot(1,3,2)
    plot(dispVar7x,dispVar7y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.255 0.069 0]','DisplayName',strcat('Bar Diam',num2str(12/8.*2.54)));
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

for j=1:nbars
    if ListRebarDiamRight(j)==4/8.*2.54
        t1=[t1,1];
        dispVar1x=[dispVar1x,distRebarRight(j,1)];
        dispVar1y=[dispVar1y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==5/8.*2.54
        t2=[t2,2];
        dispVar2x=[dispVar2x,distRebarRight(j,1)];
        dispVar2y=[dispVar2y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==6/8.*2.54
        t3=[t3,3];
        dispVar3x=[dispVar3x,distRebarRight(j,1)];
        dispVar3y=[dispVar3y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==8/8.*2.54
        t4=[t4,4];
        dispVar4x=[dispVar4x,distRebarRight(j,1)];
        dispVar4y=[dispVar4y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==9/8.*2.54
        t5=[t5,5];
        dispVar5x=[dispVar5x,distRebarRight(j,1)];
        dispVar5y=[dispVar5y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==10/8.*2.54
        t6=[t6,6];
        dispVar6x=[dispVar6x,distRebarRight(j,1)];
        dispVar6y=[dispVar6y,distRebarRight(j,2)];
    elseif ListRebarDiamRight(j)==12/8.*2.54
        t7=[t7,7];
        dispVar7x=[dispVar7x,distRebarRight(j,1)];
        dispVar7y=[dispVar7y,distRebarRight(j,2)];
    end
end
figure(6)
subplot(1,3,3)
plot(x,y,'k -','linewidth',1)
hold on
xlabel('x´')
ylabel('y´')
title('Rectangular reinforced beam (Right section)')
legend('Beam´s boundaries')
axis([-(b+10) b+10 -(h+10) h+10])
hold on

if isempty(t1)~=1
    figure(6)
    subplot(1,3,3)
    plot(dispVar1x,dispVar1y,'r o','linewidth',1,'MarkerFaceColor','red',...
        'DisplayName',strcat('Bar Diam',num2str(4/8.*2.54)));
end
if isempty(t2)~=1
    figure(6)
    subplot(1,3,3)
    plot(dispVar2x,dispVar2y,'b o','linewidth',1,'MarkerFaceColor','blue',...
        'DisplayName',strcat('Bar Diam',num2str(5/8.*2.54)));
end
if isempty(t3)~=1
    figure(6)
    subplot(1,3,3)
    plot(dispVar3x,dispVar3y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.05 0.205 0.05]','DisplayName',strcat('Bar Diam',num2str(6/8.*2.54)));
end
if isempty(t4)~=1
    figure(6)
    subplot(1,3,3)
    plot(dispVar4x,dispVar4y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.072 0.061 0.139]','DisplayName',strcat('Bar Diam',num2str(8/8.*2.54)));
end
if isempty(t5)~=1
    figure(6)
    subplot(1,3,3)
    plot(dispVar5x,dispVar5y,'k o','linewidth',1,'MarkerFaceColor',...
        'black','DisplayName',strcat('Bar Diam',num2str(9/8.*2.54)));
end
if isempty(t6)~=1
    figure(6)
    subplot(1,3,3)
    plot(dispVar6x,dispVar6y,'m o','linewidth',1,'MarkerFaceColor',...
        'magenta','DisplayName',strcat('Bar Diam',num2str(10/8.*2.54)));
end
if isempty(t7)~=1
    figure(6)
    subplot(1,3,3)
    plot(dispVar7x,dispVar7y,'o','linewidth',1,'MarkerFaceColor',...
        '[0.255 0.069 0]','DisplayName',strcat('Bar Diam',num2str(12/8.*2.54)));
end
