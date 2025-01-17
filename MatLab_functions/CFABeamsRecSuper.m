function [UNBS,UNDS,UCS,BSS,CFAS,BS,CFA]=CFABeamsRecSuper(nb6l,nb6m,nb6r,...
    dbl6,dbm6,dbr6,nbcut6l,nbcut6m,nbcut6r,Wunb,Wnd,Wcutslay)

%% Uniformity of Number of Rebars along the beam's span
% Right section
nb3sec=[nb6l;
        nb6m;
        nb6r];
    
nbcuts3sec=[nbcut6l;
            nbcut6m;
            nbcut6r];

db3sec=[dbl6;
        dbm6;
        dbr6];

for i=1:3
    nb6s=nb3sec(i,:);
    nlays=sum((nb6s~=0));
    nlays=ceil(nlays/2);
    UNBL=zeros(nlays-1,1);
    for j=1:nlays-1
        j1=j*2-1;
        j2=j*2;
        if 2<=(nb6s(j1+2)+nb6s(j2+2)) && (nb6s(j1+2)+nb6s(j2+2))<=(nb6s(1)+nb6s(2)-1)
            UNBL(j)=(nb6s(j1+2)+nb6s(j2+2))/(nb6s(1)+nb6s(2));
        else
            UNBL(j)=1;
        end
    end
    if nlays>1
        UNBS(i)=1/(nlays-1)^Wunb*sum(UNBL);
    else
        UNBS(i)=1;
    end
end

UNB=sum(UNBS)/3;

%% Number of diameters sizes
for i=1:3
    dbsec=db3sec(i,:);
    nb6s=nb3sec(i,:);
    nlays=ceil(sum((nb6s~=0))/2);
    
    % Quantifying the number of different diameter sizes per section
    dbsecdif0=[];
    nb6dif0=[];
    for ii=1:6
        if nb6s(ii)~=0 
            dbsecdif0=[dbsecdif0,dbsec(ii)];
            nb6dif0=[nb6dif0,nb6s(ii)];
        end
    end
    [dbsort,ind]=sort(dbsecdif0);
    nbsort=nb6dif0(ind);
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
    NDS=length(a);
    
    UNDS(i)=1/NDS^Wnd;
end
UND=sum(UNDS)/3;

%% Number of cuts
for i=1:3
    nb6s=nb3sec(i,:);
    nbcuts=nbcuts3sec(i,:);
    nlays=sum((nbcuts~=0));

    if nlays~=0

        nlays=ceil(nlays/2);
        UCL=zeros(nlays,1);
        for j=1:nlays
            j1=j*2-1;
            j2=j*2;
            if (nb6s(j1)+nb6s(j2))~=0
                UCL(j)=(((nb6s(j1)+nb6s(j2))-(nbcuts(j1)+nbcuts(j2)))/...
                            (nb6s(j1)+nb6s(j2)))^0.5;
            end
        end
        UCS(i)=1/nlays^Wcutslay*sum(UCL);
    else
        UCS(i)=1;
    end
end

UC=sum(UCS)/3;
    
%% Buildability Score
% Per section
BSS=UNBS+UNDS+UCS;

% Per length
BS=UNB+UND+UC;

%% Constructability Factor of Assembly
% Per section
CFAS=BSS./3;

% Per length
CFA=BS/3;
