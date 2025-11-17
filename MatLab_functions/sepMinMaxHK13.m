function [sepMin,sepMax]=sepMinMaxHK13(dbl,hagg,horver)

if horver==0 % horizontal separation
    sepMin=ceil(max([20,dbl,hagg+5]));

    sepMax=150;
elseif horver==1 % vertical separation (for beams)
    sepMin=ceil(max([20,dbl,2*hagg/3]));
    sepMax=250;
end