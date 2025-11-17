function [CS,CS1,CS2]=CSRebarMultiColsRec(D,Dmin,NBLC,NBUC,NBmaxX,NBmaxY,...
                                           WND,WNB,WDS,WNC)

%% Number of Different Diameters
NDXY = difDiamSizesLayers(D(2:3));
ND = difDiamSizesLayers(D);
CSND=(1/ND);

%% Number of Rebars

NR=2*NBLC(2)+2*NBLC(3);
NRmax=2*NBmaxX+2*NBmaxY;

CSNB=(1-min((NR/NRmax),1));

%% Cutting/Bending
% Diameter to cut and/or bend
CDS=(Dmin/(max(D)))^0.5;

% Number of rebars to cut
NBX2C=NBLC(2)-NBUC(2);
NBY2C=NBLC(3)-NBUC(3);
if sum(NBLC(2:3))==0
    CSNC=1;
else
    CSNC=1-(NBX2C+NBY2C)/(NBLC(2)+NBLC(3));
end

% Irregularity of cutting lengths
NDCut=0;
if all([NBX2C>0,NBY2C>0])
    if NDXY==2
        NDCut=2;
    else
        NDCut=1;
    end
elseif all([NBX2C==0,NBY2C>0])
        NDCut=1;
elseif all([NBX2C>0,NBY2C==0])
    NDCut=1;
elseif all([NBX2C==0,NBY2C==0])
    NDCut=0;
end

if NDCut>0
    CSDC=(1/NDCut)^0.5;
    CS1=1/3*(CDS^WDS(1)+CSNC^WNC(1)+CSDC);
else
    CS1=CDS^WDS(1); % if there are no cuts, then only the diameter is considered 
             % to account for the complexity of bending and assembly
end

%% Assembly, placing
CS2=1/2*(CSNB^WNB(1)+CSND^WND(1));

%% Constructability Score
CS=CS1+CS2;

CS=CS/2;
end

function nDiams = difDiamSizesLayers(vector)
    nitemsVec=length(vector);
    dbsecdif0=[];
    for ii=1:nitemsVec
        dbsecdif0=[dbsecdif0,vector(1,ii)];
    end
    [dbsort,ind]=sort(dbsecdif0);
    ndbdif0=length(dbsort);
    nbj=1; 
    jj=1;
    a=[];
    for ii=1:ndbdif0-1
        if dbsort(ii)~=dbsort(ii+1)
            a=[a,nbj]; 
            nbj=1; 
            jj=jj+1;
        else
            nbj=nbj+1;
        end
    end
    a=[a,nbj];
    nDiams=length(a);
end