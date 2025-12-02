function [LenRebar1,LenRebar2,LenRebar3,LenRebar4,LenRebar5,LenRebar6]=...
            lenRebarRecBeamCuts(nb6,nbAfterCut6,span,ldL6)

nbl1=nb6(1);
nbl2=nb6(2);
nbl3=nb6(3);
nbl4=nb6(4);
nbl5=nb6(5);
nbl6=nb6(6);

%% Redistribution
%% Layer 1
% Ends
LenRebar1=zeros(2,1)+span;

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

LenRebar2=zeros(nbl2,1)+ldL6(2);
LenRebar2(indecesKeep,1)=span;

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

LenRebar3=zeros(nbl3,1)+ldL6(3);
LenRebar3(indecesKeep,1)=span;

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
LenRebar4=zeros(nbl4,1)+ldL6(4);
LenRebar4(indecesKeep,1)=span;

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

LenRebar5=zeros(nbl5,1)+ldL6(5);
LenRebar5(indecesKeep,1)=span;

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

LenRebar6=zeros(nbl6,1)+ldL6(6);
LenRebar6(indecesKeep,1)=span;

