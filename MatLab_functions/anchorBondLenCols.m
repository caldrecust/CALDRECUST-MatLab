function Ldb=anchorBondLenCols(fcu,db) 

    % Table 8.4 of the code (CoP HK 2013):
    if fcu==30              
        phi=40;         
    elseif fcu==35
        phi=38;   
    elseif fcu==40
        phi=35;  
    elseif fcu==45
        phi=33;   
    elseif fcu==50
        phi=31;    
    elseif fcu>=60
        phi=28;
    end
    
    Ldb=db.*phi;
end