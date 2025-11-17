function [dispositionRebar,barDiamList,sephor1,sepver1]=distrRebarSym3Cols...
                                    (b,h,rec,nb,nb2,nb3,db1,db2,db3,dvs)

bprima=b-2*rec(1)-2*dvs;
hprima=h-2*rec(2)-2*dvs;

dispositionRebar=zeros(nb,2);

%% Bars - corners

xl1=linspace(-0.5*bprima,0.5*bprima,2);
dispositionRebar(1:2,1)=xl1';
dispositionRebar(1:2,2)=0.5*hprima-0.5*db1;

dispositionRebar(3:4,1)=xl1';
dispositionRebar(3:4,2)=-0.5*hprima+0.5*db1;

%% Bars - top and bottom
sephor1=round((bprima-2*db1-nb2*db2)/(2+nb2-1),1);

xl2=linspace(-0.5*bprima+db1+sephor1,0.5*bprima-db1-sephor1,nb2);
dispositionRebar(5:4+nb2,1)=xl2';
dispositionRebar(5:4+nb2,2)=0.5*hprima-0.5*db1;

dispositionRebar(4+nb2+1:4+2*nb2,1)=xl2';
dispositionRebar(4+nb2+1:4+2*nb2,2)=-0.5*hprima+0.5*db1;


%% Bars - sides

sepver1=round((hprima-2*db1-nb3*db3)/(2+nb3-1),1);
yl3=linspace(-0.5*hprima+db1+sepver1,0.5*hprima-db1-sepver1,nb3);
dispositionRebar(4+2*nb2+1:4+2*nb2+nb3,2)=yl3';
dispositionRebar(4+2*nb2+1:4+2*nb2+nb3,1)=-0.5*bprima+0.5*db3;

dispositionRebar(4+2*nb2+nb3+1:4+2*nb2+2*nb3,2)=yl3';
dispositionRebar(4+2*nb2+nb3+1:4+2*nb2+2*nb3,1)=0.5*bprima-0.5*db3;

%% List of rebar diameter
% Order is considered, with respect of rebar distribution matrix
barDiamList(1:4,1)=db1;
barDiamList(4+1:4+2*nb2,1)=db2;
barDiamList(4+2*nb2+1:nb,1)=db3;