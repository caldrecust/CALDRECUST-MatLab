function [nb1r,nb2r,nb3r,isfeasible]=nb1DSec2(nbmaxr,Aos,ab,nb3l,nbAfterCut3)


nbro=ceil(Aos/ab); % minimum required number of bars for left cross-section

if nbro<2
    nbro=2;
end

isfeasible=false;
if nbro>3*nbmaxr
    isfeasible=false;
    nb1r=0;
    nb2r=0;
    nb3r=0;
else
    isfeasible=true;
    if sum(nbAfterCut3)>=nbro % no more rebars are needed
        nb1r=nbAfterCut3(1);
        nb2r=nbAfterCut3(2);
        nb3r=nbAfterCut3(3);
        
    else
        if nbro<=nbmaxr 
            if nbAfterCut3(1)==2
                nb1r=nbro;
                if nb1r<nbAfterCut3(1)
                    nb1r=2;
                end
                nb2r=0;
                nb3r=0;
                
            elseif nbAfterCut3(1)>2
                nb1r=nb3l(1);
                
                if nbro-nb1r>0
                    nb2r=nbro-nb1r;
                else
                    nb2r=0;
                end
                
                if nb2r<nbAfterCut3(2)
                    nb2r=nbAfterCut3(2);
                end
                
                nb3r=0;
                
                if nb3r<nbAfterCut3(3)
                    nb3r=nbAfterCut3(3);
                end
            end
            if mod(nb1r,2)==0 && mod(nb2r,2)~=0
                nb2r=nb2r+1;
            end
            
            if mod(nb2r,2)==0 && mod(nb3r,2)~=0
                nb3r=nb3r-1;
            end
            
        elseif nbro>nbmaxr
            nb1r=nb3l(1);
            if nbro-nb1r>=nbmaxr
                nb2r=nb3l(1);
                nb3r=nbro-nb1r-nb2r;
                if nb3r<nbAfterCut3(3)
                    nb3r=nbAfterCut3(3);
                end
                if mod(nb2r,2)==0 && mod(nb3r,2)~=0
                    nb3r=nb3r+1;
                end
                
            elseif nbro-nb1r<nbmaxr
                nb2r=nbro-nb1r;
                if nb2r<nbAfterCut3(2)
                    nb2r=nbAfterCut3(2);
                end
                nb3r=0;
                if mod(nb1r,2)==0 && mod(nb2r,2)~=0
                    nb2r=nb2r+1;
                end
                
            end
        end
    end
end
