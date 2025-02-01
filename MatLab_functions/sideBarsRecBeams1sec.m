function [dSb,nSb,sepSb,dtb,distrSideBars]=sideBarsRecBeams1sec(b,h,fy,brec,...
    hrec,nb6,db6,dstirrup,minVsep,rebarAvailable)

bp=h-2*brec-2*dstirrup-db6(1);

%% Number of rebar layers in tension
nlay=ceil(sum((nb6~=0))/2);

%% Number of side rebars 

dtb=hrec+dstirrup+(nlay-1)*minVsep+nlay*max(db6);

hSb=2*h/3; % efective height distance in which side rebars will be 
           % distributed
maxSb=250; % max separation of side rebars
nSb=ceil((hSb-dtb)/maxSb); % number of side rebars 

%% Separation of side rebars
sepSb=(hSb-dtb)/nSb;

%% Diameter of side rebars
dminSb=sqrt(sepSb*b/fy);

nbarCommercial=length(rebarAvailable(:,1));
for i=1:nbarCommercial
    if rebarAvailable(i,2)>=dminSb
        break;
    end
end
dSb=rebarAvailable(i,2);

%% Distribution of bars as local coordinates

% Effective height
lowH=-0.5*h+dtb+sepSb;
upH=hSb-0.5*h; 

%% Separation of side rebars

ysb=linspace(lowH,upH,nSb);
xsb=[-0.5*bp,0.5*bp];

for i=1:nSb
    sxysb(i,:)=[xsb(1),ysb(i)];
    sxysb(nSb+i,:)=[xsb(2),ysb(i)];
end
distrSideBars=sxysb;