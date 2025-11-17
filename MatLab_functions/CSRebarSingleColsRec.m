function [CS,CS1,CS2]=CSRebarSingleColsRec(D,Dmin,NBX,NBY,NBmaxX,NBmaxY,...
                                           WND,WNB,WDS,WNC)

%% Number of Different Diameters
NDXY = difDiamSizesLayers(D(2:3));
ND = difDiamSizesLayers(D);
CSND=(1/ND);

%% Number of Rebars
NR=2*NBX+2*NBY;
NRmax=2*NBmaxX+2*NBmaxY;

CSNB=(1-min((NR/NRmax),1));

%% Cutting/Bending
CDS=(Dmin/(max(D)))^0.5;


% Number of rebars to cut

CSNC=0;

% Irregularity of cutting lengths
NDCut=ND;

CSDC=(1/NDCut)^0.5;
CS1=1/3*(CDS^WDS(1)+CSNC^WNC(1)+CSDC);


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