function [nb1,nb2,nb3,isfeasible]=nb3DSec2(nbmax,Aos,ab,nb3l,nbAfterCut3)
    
    if nb3l(1)==2
        Abmax1 = nbmax(1) * ab(1) ;
        Abmax2 = nbmax(2) * ab(2) ;
        Abmax3 = nbmax(3) * ab(3) ;
    else
        Abmax1 = nb3l(1) * ab(1) ;
        Abmax2 = nb3l(1) * ab(2) ;
        Abmax3 = nb3l(1) * ab(3) ;
    end
    if sum(nbAfterCut3.*ab)>=Aos % no more rebar is required
        isfeasible=true;
        nb1=nbAfterCut3(1);
        nb2=nbAfterCut3(2);
        nb3=nbAfterCut3(3);
    else
        if Abmax1 >= Aos % only one rebar layer is necessary
            isfeasible=true;
            if nbAfterCut3(1)==2
                
                nb1 = min(ceil(Aos / ab(1)),nbmax(1)) ;
                nb2=0;
                nb3=0;
            elseif nbAfterCut3(1)>2
                nb1=min(ceil(Aos / ab(1)),nbAfterCut3(1)) ;
                Ab1=nb1*ab(1);
                if Aos-Ab1>=0
                    nb2=ceil((Aos-Ab1)/(ab(2)));
                    if nb2>nb1
                        nb2=nb1;
                    end
                else
                    nb2 = nbAfterCut3(2);
                end
                
                nb3 = nbAfterCut3(3);
            end
            
            if mod(nb1,2)==0 && mod(nb2,2)~=0
                if any([nb2==nb1,nb2==nbmax(2)])
                    nb2=nb2-1;
                    nb3=nb3+1;
                else
                    nb2=nb2+1;
                    nb3=0;
                end
            end
            
            if mod(nb2,2)==0 && mod(nb3,2)~=0
                nb3=nb3+1;
            end
        elseif any([all([Abmax1 < Aos, Abmax1 + Abmax2 >= Aos]),...
                   all([Abmax1 + Abmax2 < Aos, Abmax1 + Abmax2 + Abmax3 >= Aos])])
            isfeasible=true;
            if nbAfterCut3(1)==2
                nb1=nbmax(1);
            else
                nb1= nb3l(1);
            end
            Ab1=nb1*ab(1);
            
            nb2=max([ceil((Aos-Ab1)/ab(2)),nbAfterCut3(2)]);
            if nb2>nb1
                nb2=nb1;
            end
            Ab2=nb2*ab(2);
            
            if Aos-Ab1-Ab2>0
                nb3=max([ceil((Aos-Ab1-Ab2)/ab(3)),nbAfterCut3(3)]);
                if nb3>nb2
                    nb3=nb2;
                end
            else
                nb3=nbAfterCut3(3);
            end
                
            if mod(nb1,2)==0 && mod(nb2,2)~=0
                nb2=nb2+1;
            end
            if mod(nb2,2)==0 && mod(nb3,2)~=0
                if any([nb3==nbmax(3),nb3==nb2])
                    nb3=nb3-1;
                else
                    nb3=nb3+1;
                end
                
            end

        elseif all([Abmax1 + Abmax2 + Abmax3 < Aos])
            isfeasible=false;
            nb1=0;
            nb2=0;
            nb3=0;
        end
    end
end