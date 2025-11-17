function [distrRebar]=distrRecBeam3tLayers(b,h,hrec,t,vSep)

dvs=10;
hp=h-2*hrec-2*dvs;
bp=b-2*hrec-2*dvs;
distrRebar=zeros(4,1);

%% Tension

distrRebar(1,1)=-0.5*hp+0.5*t(1);

distrRebar(2,1)=-0.5*hp+t(1)+vSep+0.5*t(2);

distrRebar(3,1)=-0.5*hp+t(1)+vSep+t(2)+vSep+0.5*t(3);

%% Compression
tmin=0.003*b*h/bp;
distrRebar(4,1)=0.5*hp-0.5*tmin;
