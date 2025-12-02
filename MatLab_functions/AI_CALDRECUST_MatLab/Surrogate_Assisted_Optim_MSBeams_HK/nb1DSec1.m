function [nb1,nb2,nb3,isfeasible]=nb1DSec1(nbmax,Aos3,ab)

    nblo=ceil(Aos3(1)/ab(1)); % minimum required number of bars for left cross-section
    if nblo<2
        nblo=2;
    end
    
    isfeasible=false;
    if nblo>3*nbmax
        nb1=0;
        nb2=0;
        nb3=0;
        isfeasible=false;
    else
        isfeasible=true;
        if nblo<=nbmax
            nb1=nblo;
            nb2=0;
            nb3=0;
            
        elseif nblo>nbmax
            nb1=nbmax;
            if nblo-nb1>nbmax
                nb2=nbmax;
                nb3=nblo-nb1-nb2;
                
                if any([mod(nb2,2)==0,mod(nb3,2)~=0])
                    nb3=nb3+1;
                end
            elseif nblo-nb1==nbmax
                nb2=nbmax;
                nb3=0;
            elseif nblo-nb1<nbmax
                nb2=nblo-nb1;
                nb3=0;
                if any([mod(nb1,2)==0,mod(nb2,2)~=0])
                    nb2=nb2+1;
                end
            end
        end
    end
end