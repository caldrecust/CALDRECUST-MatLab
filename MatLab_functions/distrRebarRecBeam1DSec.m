function [distrRebar,listRebarDiams]=distrRebarRecBeam1DSec(nb3,db,b,h,...
                                                      brec,hrec,vSep)

dvs=10;

hp=h-2*hrec-2*dvs-db(1);
bp=b-2*brec-2*dvs-[db];

nbl1=nb3(1);
nbl2=nb3(2);
nbl3=nb3(3);

dbl1=db(1);

nb=sum(nb3); % total number of rebars in tension
distrRebar=zeros(nb,2);

%% Conventional distribution
%% Layer 1

xl1=linspace(-0.5*bp,0.5*bp,nbl1);
distrRebar(1:nbl1,1)=xl1';
distrRebar(1:nbl1,2)=-0.5*hp+0.5*dbl1;

listRebarDiams(1:nbl1,1)=dbl1;

%% Layer 2

if nbl2>1
    xl2=linspace(-0.5*bp,0.5*bp,nbl2);
    distrRebar(nbl1+1:nbl1+nbl2,1)=xl2';
    distrRebar(nbl1+1:nbl1+nbl2,2)=-0.5*hp+dbl1+...
                                   vSep+0.5*dbl1;
elseif nbl2==1
    distrRebar(nbl1+1:nbl1+nbl2,1)=0;
    distrRebar(nbl1+1:nbl1+nbl2,2)=-0.5*hp+dbl1+...
                                    vSep+0.5*dbl1;
end
listRebarDiams(nbl1+1:nbl1+nbl2,1)=dbl1;

%% Layer 3

if nbl3>1
    xl2=linspace(-0.5*bp,0.5*bp,nbl3);
    distrRebar(nbl1+nbl2+1:nbl1+nbl2+nbl3,1)=xl2';
    distrRebar(nbl1+nbl2+1:nbl1+nbl2+nbl3,2)=-0.5*hp+dbl1+...
                                              vSep+dbl1+...
                                              vSep+0.5*dbl1;
elseif nbl3==1
    distrRebar(nbl1+nbl2+1:nbl1+nbl2+nbl3,1)=0;
    distrRebar(nbl1+nbl2+1:nbl1+nbl2+nbl3,2)=-0.5*hp+dbl1+...
                                              vSep+dbl1+...
                                              vSep+0.5*dbl1;
end
listRebarDiams(nbl1+nbl2+1:nbl1+nbl2+nbl3,1)=dbl1;