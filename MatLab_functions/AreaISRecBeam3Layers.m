function As=AreaISRecBeam3Layers(t,b,Cc)

dvs=10;
bp=b-2*Cc-2*dvs;

As=sum(t.*bp);
end