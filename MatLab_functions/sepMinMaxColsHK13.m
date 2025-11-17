function [sepMin,sepMax]=sepMinMaxColsHK13(dbl,hagg)

    sepMin=ceil(max([20,min(dbl),hagg+5]));

    sepMax=min([12*min(dbl),200]);
end