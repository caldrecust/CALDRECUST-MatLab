function [nbnew1,nbnew2,nbnew3,Abcut]=BendCutRebarSecBeamAs1DiamLayer(nb3,db3,...
                        Ast,b,h,hrec,vSep)
   
Amin=0.003*b*h;

ab3=pi/4*db3.^2;
Ab3=nb3.*ab3;
Abtotal=sum(Ab3);

d=h-hrec;
if nb3(2)>0
    d=d-vSep-db3(2);
end
if nb3(3)>0
    d=d-vSep-db3(3);
end

Abm=max([Ast,Amin]);

Abtarget=Abtotal-Abm;
if Abtarget>0.5*Abtotal % amount of rebar area to be bent. Note: no more 
                        % than 50% of the total current rebar can be cut
    Abtarget=0.5*Abtotal;
end
if Ab3(3)>Abtarget % only rebars on the outer layer will be bent
    
    nbcut3=fix(Abtarget/ab3(3)); % number of rebar in outer layer
    
    if nb3(3)>=nbcut3
        if mod(nb3(3),2)==0
            if mod(nbcut3,2)~=0
                nbcut3=nbcut3-1;
            end
        elseif mod(nb3(3),2)~=0
            if mod(nbcut3,2)==0
                nbcut3=nbcut3-1;
            end
        end
        Abcut=nbcut3*ab3(3);
        nbnew1=nb3(1); nbnew2=nb3(2); nbnew3=nb3(3)-nbcut3;
    else
        Abcut=nb3(3)*ab3(3); % otherwise, all mid rebars of 
                             % third layer will be cut
        nbnew1=nb3(1); nbnew2=nb3(2); nbnew3=0;
    end
    
elseif Ab3(3)+Ab3(2)>Abtarget % rebar on the mid layer 
                                     % will be cut

    Abcut2=Abtarget-Ab3(3);
    nbcut2=fix(Abcut2/ab3(2));
    if mod(nb3(2),2)==0
        if mod(nbcut2,2)~=0
            nbcut2=nbcut2-1;
        end
    else
        if mod(nbcut2,2)==0 && nbcut2~=0
            nbcut2=nbcut2-1;
        end
    end
    Abcut=Ab3(3)+nbcut2*ab3(2);
    
    nbnew1=nb3(1); nbnew2=nb3(2)-nbcut2; nbnew3=0;
    
else % rebars on the three layers will be cut
    Abcut1=Abtarget-Ab3(3)-Ab3(2);
    nbcut1=fix(Abcut1/ab3(1));
    if nb3(1)-2>=nbcut1
        if mod(nb3(1),2)==0
            if mod(nbcut1,2)~=0
                nbcut1=nbcut1-1;
            end
        else
            if mod(nbcut1,2)==0 && nbcut1~=0
                nbcut1=nbcut1-1;
            end
        end
        Abcut=Ab3(3)+Ab3(2)+nbcut1*ab3(1);
        
        nbnew1=nb3(1)-nbcut1; nbnew2=0; nbnew3=0; 
    else
        Abcut=Ab3(3)+Ab3(2)+(nb3(1)-2)*ab3(1);
        nbnew1=2; nbnew2=0; nbnew3=0; 
    end
    
end