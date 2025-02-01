function [dSb,nSb,sepSb,distrSideBars]=sideBarsRecBeams3SecSpan(b,h,fy,...
            brec,hrec,nb18Spans,db18Spans,dstirrup,hagg,rebarAvailable)

%% Number of rebar layers in tension
nb6L=nb18Spans(1:6);
nb6M=nb18Spans(7:12);
nb6R=nb18Spans(13:18);

nlayL=ceil(sum((nb6L~=0))/2);
nlayM=ceil(sum((nb6M~=0))/2);
nlayR=ceil(sum((nb6R~=0))/2);

db6L=db18Spans(1,1:6);
db6M=db18Spans(1,7:12);
db6R=db18Spans(1,13:18);

% Left section
[minVsepL,maxVsep]=sepMinMaxHK13(db6L,hagg,1);

dtbL=hrec+dstirrup+(nlayL-1)*minVsepL+nlayL*max(db6L);

% Mid section
[minVsepM,maxVsep]=sepMinMaxHK13(db6M,hagg,1);

dtbM=hrec+dstirrup+(nlayM-1)*minVsepM+nlayM*max(db6M);

% Right section
[minVsepR,maxVsep]=sepMinMaxHK13(db6R,hagg,1);

dtbR=hrec+dstirrup+(nlayR-1)*minVsepR+nlayR*max(db6R);
    
%% Local coordinate distribution

% Effective height
lowH=-0.5*h+dtbM+minVsepM;
upH=0.5*h-max([dtbL,dtbR])-max([minVsepL,minVsepR]);

he=upH-lowH;
nSb=ceil((he)/maxVsep)-1; % number of side rebars 

%% Separation of side rebars
sepSb=(he)/(nSb+1);

%% Diameter of side rebars
dminSb=sqrt(sepSb*b/fy);
nbarCommercial=length(rebarAvailable(:,1));
for i=1:nbarCommercial
    if rebarAvailable(i,2)>=dminSb
        break;
    end
end
dSb=rebarAvailable(i,2);

bp=b-2*brec-2*dstirrup-dSb;

% Distribution of rebars in local coordinates
ysb=linspace(lowH+sepSb,upH-sepSb,nSb);
xsb=[-0.5*bp,0.5*bp];

for i=1:nSb
    distrSideBars(i,:)=[xsb(1),ysb(i)];
    distrSideBars(nSb+i,:)=[xsb(2),ysb(i)];
end

