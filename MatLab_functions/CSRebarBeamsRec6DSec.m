function [UNBS,UNDS,UCS,BSS,CFAS,BS,CFA]=CSRebarBeamsRec6DSec(nb6l,nb6m,nb6r,...
    dbl6,dbm6,dbr6,nbcut6l,nbcut6m,nbcut6r,Wunb,Wnd,Wcut)
    
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
                UNBL(j)=((nb6s(j1+2)+nb6s(j2+2))/(nb6s(1)+nb6s(2)))^Wunb(2);
            else
                UNBL(j)=1;
            end
        end
        if nlays>1
            UNBS(i)=1/(nlays-1)^Wunb(1)*sum(UNBL);
        else
            UNBS(i)=1;
        end
    end

    UNB=sum(UNBS)/3;

    %% Number of diameters sizes
    for i=1:3
        dbsec=db3sec(i,:);
        nb6s=nb3sec(i,:);

        % Quantifying the number of different diameter sizes per section
        NDS = difDiamSizesLayers(dbsec,nb6s);

        UNDS(i)=1/NDS^Wnd(1);
    end
    UND=sum(UNDS)/3;

    %% Number of cuts
    for i=1:3
        nb6s=nb3sec(i,:);
        nbcuts=nbcuts3sec(i,:);
        nlaysCut=sum((nbcuts~=0));

        if nlaysCut~=0

            nlaysCut=ceil(nlaysCut/2);
            UCL=zeros(nlaysCut,1);
            for j=1:nlaysCut
                j1=j*2-1;
                j2=j*2;
                if (nb6s(j1)+nb6s(j2))~=0
                    UCL(j)=(((nb6s(j1)+nb6s(j2))-(nbcuts(j1)+nbcuts(j2)))/...
                                (nb6s(j1)+nb6s(j2)))^Wcut(2);
                end
            end
            UCS(i)=1/nlaysCut^Wcut(1)*sum(UCL);
        else
            UCS(i)=1;
        end
    end

    UC=sum(UCS)/3;

    %% Buildability Score
    % Per section
    BSS=UNBS.^Wunb(3)+UNDS.^Wnd(2)+UCS.^Wcut(3);

    % Per length
    BS=UNB^Wunb(3)+UND^Wnd(2)+UC^Wcut(3);

    %% Constructability Factor of Assembly
    % Per section
    CFAS=BSS./3;

    % Per length
    CFA=BS/3;
end

function nItems = difDiamSizesLayers(vector,refDifZero)
    nitemsVec=length(vector);
    dbsecdif0=[];
    for ii=1:nitemsVec
        if refDifZero(ii)~=0
            dbsecdif0=[dbsecdif0,vector(1,ii)];
        end
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
    nItems=length(a);
end
