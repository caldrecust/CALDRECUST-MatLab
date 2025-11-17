function [UNBS,UNDS,UCS,BSS,CFAS,BS,CFA]=CSRebarBeamsRec1DSec(nb3l,nb3m,nb3r,...
    dbl,dbm,dbr,nbcut3l,nbcut3m,nbcut3r,Wunb,Wnd,Wcut)

    %% Uniformity of Number of Rebars along the beam's span
    % Right section
    nb3sec=[nb3l;
            nb3m;
            nb3r];

    nbcuts3sec=[nbcut3l;
                nbcut3m;
                nbcut3r];
    db3sec=[dbl;
            dbm;
            dbr];

    for i=1:3
        nb3s=nb3sec(i,:);
        nlays=sum((nb3s~=0));
        nlays=ceil(nlays);
        UNBL=zeros(nlays-1,1);
        for j=1:nlays-1
            if nb3s(j+1)<=(nb3s(1))
                UNBL(j)=(nb3s(j+1)/(nb3s(1)))^Wunb(2);
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
    for i=1:1
        dbsec=db3sec(:,i);
        nb3s=nb3sec(:,i);

        % Quantifying the number of different diameter sizes per section
        NDS = difDiamSizesLayers(dbsec',nb3s');

        UNDS(i)=1/NDS^Wnd(1);
    end
    UND=UNDS;
    
    %% Number of cuts
    for i=1:3
        nb3s=nb3sec(i,:);
        nbcuts=nbcuts3sec(i,:);
        nlaysCut=sum((nbcuts~=0));

        if nlaysCut~=0

            nlaysCut=ceil(nlaysCut);
            UCL=zeros(nlaysCut,1);
            for j=1:nlaysCut
                if (nb3s(j)~=0)
                    UCL(j)=((nb3s(j)-nbcuts(j))/...
                                (nb3s(j)))^Wcut(2);
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


function nDiams = difDiamSizesLayers(vector,refDifZero)
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
    nDiams=length(a);
end