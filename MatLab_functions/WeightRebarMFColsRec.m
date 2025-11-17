function WsRebarCol=WeightRebarMFColsRec(nb3UCol,nb3LCol,db3,ws,Hcol,fcu)
    
    AsUCol=nb3UCol.*db3.^2*pi/4;
    
    nb2Cut=nb3UCol-nb3LCol;
    As2Cut=nb2Cut.*db3.^2*pi/4;
    
    Ldb=anchorBondLenCols(fcu,db3);
    
    Wnb2Cut=sum(As2Cut.*(0.5*Hcol+Ldb)*ws);
    Wnbkeep=sum(AsUCol*Hcol*ws);
    
    WsRebarCol=Wnbkeep+Wnb2Cut;
    
end