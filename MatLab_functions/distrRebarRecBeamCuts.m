function [distrRebarNew,listRebarDiamsNew]=distrRebarRecBeamCuts...
                            (nb6,db6,b,h,brec,hrec,vSep,nbAfterCut6)

global dvs

hp=h-2*hrec-2*dvs-max(db6(1:2));
bp=b-2*brec-2*dvs-max([db6(1),db6(3),db6(5)]);

nbl1=nb6(1);
nbl2=nb6(2);
nbl3=nb6(3);
nbl4=nb6(4);
nbl5=nb6(5);
nbl6=nb6(6);

dbl1=db6(1);
dbl2=db6(2);
dbl3=db6(3);
dbl4=db6(4);
dbl5=db6(5);
dbl6=db6(6);

nb=sum(nb6); % total number of rebars in tension
distrRebar=zeros(nb,2);

sepLowLay=(bp)/((nbl1+nbl2)-1); % rebar separation - center to center

%% Conventional distribution
%% Layer 1
% Ends

xl1=linspace(-0.5*bp,0.5*bp,nbl1);
distrRebar(1:nbl1,1)=xl1';
distrRebar(1:nbl1,2)=-0.5*hp+0.5*max(dbl1,dbl2);

listRebarDiams(1:nbl1,1)=dbl1;

% Middle
bp2=bp-2*dbl1-2*sepLowLay;

if nbl2>1
    xl2=linspace(-0.5*bp2,0.5*bp2,nbl2);
    distrRebar(nbl1+1:nbl1+nbl2,1)=xl2';
    distrRebar(nbl1+1:nbl1+nbl2,2)=-0.5*hp+0.5*max(dbl1,dbl2);
elseif nbl2==1
    distrRebar(nbl1+1:nbl1+nbl2,1)=0;
    distrRebar(nbl1+1:nbl1+nbl2,2)=-0.5*hp+0.5*max(dbl1,dbl2);
end

listRebarDiams(nbl1+1:nbl1+nbl2,1)=dbl2;

%% Layer 2
% Ends
xl1=linspace(-0.5*bp,0.5*bp,nbl3);
distrRebar(nbl1+nbl2+1:nbl1+nbl2+nbl3,1)=xl1';
distrRebar(nbl1+nbl2+1:nbl1+nbl2+nbl3,2)=-0.5*hp+max(dbl1,dbl2)+vSep+...
                                            0.5*max(dbl3,dbl4);

listRebarDiams(nbl1+nbl2+1:nbl1+nbl2+nbl3,1)=dbl3;
msepLay=(nbl2-1)/(nbl4-1);
% Middle
bp2=bp-2*dbl3-2*sepLowLay*msepLay;
if nbl4>1
    xl2=linspace(-0.5*bp2,0.5*bp2,nbl4);
    distrRebar(nbl1+nbl2+nbl3+1:nbl1+nbl2+nbl3+nbl4,1)=xl2';
    distrRebar(nbl1+nbl2+nbl3+1:nbl1+nbl2+nbl3+nbl4,2)=-0.5*hp+max(dbl1,dbl2)+...
                                                        vSep+0.5*max(dbl3,dbl4);
elseif nbl4==1
    distrRebar(nbl1+nbl2+nbl3+1:nbl1+nbl2+nbl3+nbl4,1)=0;
    distrRebar(nbl1+nbl2+nbl3+1:nbl1+nbl2+nbl3+nbl4,2)=-0.5*hp+max(dbl1,dbl2)+...
                                                        vSep+0.5*max(dbl3,dbl4);
end
listRebarDiams(nbl1+nbl2+nbl3+1:nbl1+nbl2+nbl3+nbl4,1)=dbl4;

%% Layer 3
% Ends
xl1=linspace(-0.5*bp,0.5*bp,nbl5);
distrRebar(nbl1+nbl2+nbl3+nbl4+1:nbl1+nbl2+nbl3+nbl4+nbl5,1)=xl1';
distrRebar(nbl1+nbl2+nbl3+nbl4+1:nbl1+nbl2+nbl3+nbl4+nbl5,2)=-0.5*hp+max(dbl1,dbl2)+...
                                                                vSep+max(dbl3,dbl4)+...
                                                                vSep+0.5*max(dbl5,dbl6);

listRebarDiams(nbl1+nbl2+nbl3+nbl4+1:nbl1+nbl2+nbl3+nbl4+nbl5,1)=dbl5;
msepLay=(nbl2-1)/(nbl6-1);
% Middle
bp2=bp-2*dbl5-2*sepLowLay*msepLay;
if nbl6>1
    xl2=linspace(-0.5*bp2,0.5*bp2,nbl6);
    distrRebar(nbl1+nbl2+nbl3+nbl4+nbl5+1:nbl1+nbl2+nbl3+nbl4+nbl5+nbl6,1)=xl2';
    distrRebar(nbl1+nbl2+nbl3+nbl4+nbl5+1:nbl1+nbl2+nbl3+nbl4+nbl5+nbl6,2)=-0.5*hp+max(dbl1,dbl2)+...
                                                                            vSep+max(dbl3,dbl4)+...
                                                                            vSep+0.5*max(dbl5,dbl6);
elseif nbl6==1
    distrRebar(nbl1+nbl2+nbl3+nbl4+nbl5+1:nbl1+nbl2+nbl3+nbl4+nbl5+nbl6,1)=0;
    distrRebar(nbl1+nbl2+nbl3+nbl4+nbl5+1:nbl1+nbl2+nbl3+nbl4+nbl5+nbl6,2)=-0.5*hp+max(dbl1,dbl2)+...
                                                                            vSep+max(dbl3,dbl4)+...
                                                                            vSep+0.5*max(dbl5,dbl6);
end
listRebarDiams(nbl1+nbl2+nbl3+nbl4+nbl5+1:nbl1+nbl2+nbl3+nbl4+nbl5+nbl6,1)=dbl6;

%% Redistribution
%% Layer 1
% Ends
distrRebarNew1(:,1)=distrRebar(1:2,1);
distrRebarNew1(:,2)=distrRebar(1:2,2);
listRebarDiamsNew1=listRebarDiams(1:2,1);

% Mid-part
nb2new=nbAfterCut6(2);

nbcut2=nbl2-nb2new;
startRemove=(nb2new)/2+1; % if cuts are to be executed, the bars in the
                          % middle outwards are removed
finishRemove=startRemove-1+nbcut2;
indecesKeep=[];
for i=1:nbl2
    if i<startRemove || i>finishRemove
        indecesKeep=[indecesKeep;i];
    end
end
indecesKeep=indecesKeep+nbl1;
distrRebarNew2(:,1)=distrRebar(indecesKeep,1);
distrRebarNew2(:,2)=distrRebar(indecesKeep,2);

listRebarDiamsNew2=listRebarDiams(indecesKeep,1);

%% Layer 2
% Ends
nb3new=nbAfterCut6(3);
nbcut3=nbl3-nb3new;
startRemove=(nb3new)/2+1;
finishRemove=startRemove-1+nbcut3;
indecesKeep=[];
for i=1:nbl3
    if i<startRemove || i>finishRemove
        indecesKeep=[indecesKeep;i];
    end
end
indecesKeep=indecesKeep+nbl1+nbl2;
distrRebarNew3(:,1)=distrRebar(indecesKeep,1);
distrRebarNew3(:,2)=distrRebar(indecesKeep,2);

listRebarDiamsNew3=listRebarDiams(indecesKeep,1);

% Mid-part
nb4new=nbAfterCut6(4);
nbcut4=nbl4-nb4new;
startRemove=(nb4new)/2+1;
finishRemove=startRemove-1+nbcut4;
indecesKeep=[];
for i=1:nbl4
    if i<startRemove || i>finishRemove
        indecesKeep=[indecesKeep;i];
    end
end
indecesKeep=indecesKeep+nbl1+nbl2+nbl3;
distrRebarNew4(:,1)=distrRebar(indecesKeep,1);
distrRebarNew4(:,2)=distrRebar(indecesKeep,2);

listRebarDiamsNew4=listRebarDiams(indecesKeep,1);

%% Layer 3
% Ends
nb5new=nbAfterCut6(5);
nbcut5=nbl5-nb5new;
startRemove=(nb5new)/2+1;
finishRemove=startRemove-1+nbcut5;
indecesKeep=[];
for i=1:nbl5
    if i<startRemove || i>finishRemove
        indecesKeep=[indecesKeep;i];
    end
end
indecesKeep=indecesKeep+nbl1+nbl2+nbl3+nbl4;
distrRebarNew5(:,1)=distrRebar(indecesKeep,1);
distrRebarNew5(:,2)=distrRebar(indecesKeep,2);

listRebarDiamsNew5=listRebarDiams(indecesKeep,1);

% Mid-part
nb6new=nbAfterCut6(6);
nbcut6=nbl6-nb6new;
startRemove=(nb6new)/2+1;
finishRemove=startRemove-1+nbcut6;
indecesKeep=[];
for i=1:nbl6
    if i<startRemove || i>finishRemove
        indecesKeep=[indecesKeep;i];
    end
end
indecesKeep=indecesKeep+nbl1+nbl2+nbl3+nbl4+nbl5;
distrRebarNew6(:,1)=distrRebar(indecesKeep,1);
distrRebarNew6(:,2)=distrRebar(indecesKeep,2);

listRebarDiamsNew6=listRebarDiams(indecesKeep,1);

%% Assembling of distributions
distrRebarNew=[distrRebarNew1;
               distrRebarNew2;
               distrRebarNew3;
               distrRebarNew4;
               distrRebarNew5;
               distrRebarNew6];
           
listRebarDiamsNew=[listRebarDiamsNew1;
                  listRebarDiamsNew2;
                  listRebarDiamsNew3;
                  listRebarDiamsNew4;
                  listRebarDiamsNew5;
                  listRebarDiamsNew6];
              