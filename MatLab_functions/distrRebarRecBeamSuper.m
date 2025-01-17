function [distrRebar,listRebarDiams]=distrRebarRecBeamSuper(nb6,db6,b,h,...
                                    brec,hrec,vSep,sepLowLay)
hp=h-2*hrec;
bp=b-2*brec;

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

nb=sum(nb6);
distrRebar=zeros(nb,2);

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

