function [nbnew1,nbnew2,nbnew3,nbnew4,nbnew5,nbnew6,Abcut]=...
                cutRebarSecBeamMr(nb6,db6,Mt,b,h,hrec,vSep,fc,fy)
            

ab6=pi/4*db6.^2;
Ab6=nb6.*ab6;
Abtotal=sum(Ab6);

d=h-hrec;
if nb6(3)+nb6(4)>0
    d=d-vSep-max(db6(3),db6(4));
end
if nb6(5)+nb6(6)>0
    d=d-vSep-max(db6(5),db6(6));
end

k=abs(Mt/(b*d^2*0.87*0.45*fc));
q=1-sqrt(1-4*(0.5*k));

Abm=b*h*0.45*fc/fy*q; % min required steel cross-section area

Abtarget=Abtotal-Abm;
if Abtarget>0.5*Abtotal % amount of rebar area to be cut. Note: no more 
                        % than 50% of the total current rebar can be cut
    Abtarget=0.5*Abtotal;
end
if Ab6(5)+Ab6(6)>Abtarget % only rebars on the outer layer will be cut
    
    nbcut6=fix(Abtarget/ab6(6)); % number of rebar in mid - outer layer
    if nb6(6)>=nbcut6
        if mod(nb6(6),2)==0
            if mod(nbcut6,2)~=0
                nbcut6=nbcut6-1;
            end
        elseif mod(nb6(6),2)~=0
            if mod(nbcut6,2)==0
                nbcut6=nbcut6-1;
            end
        end
        Abcut=nbcut6*ab6(6);
        nbnew1=nb6(1); nbnew2=nb6(2); nbnew3=nb6(3); 
        nbnew4=nb6(4); nbnew5=nb6(5); nbnew6=nb6(6)-nbcut6;
    else
        Abcut=nb6(6)*ab6(6)+nb6(5)*ab6(5); % otherwise, all mid rebars of 
                                           % third layer will be cut
        nbnew1=nb6(1); nbnew2=nb6(2); nbnew3=nb6(3); 
        nbnew4=nb6(4); nbnew5=nb6(5); nbnew6=0;
    end
    
elseif Ab6(5)+Ab6(6)+Ab6(4)>Abtarget % rebar on the mid part of mid layer 
                                    % will be cut

    Abcut4=Abtarget-Ab6(5)-Ab6(6);
    nbcut4=fix(Abcut4/ab6(4));
    if mod(nb6(4),2)==0
        if mod(nbcut4,2)~=0
            nbcut4=nbcut4-1;
        end
    else
        if mod(nbcut4,2)==0 && nbcut4~=0
            nbcut4=nbcut4-1;
        end
    end
    Abcut=Ab6(5)+Ab6(6)+nbcut4*ab6(4);
    
    nbnew1=nb6(1); nbnew2=nb6(2); nbnew3=nb6(3); 
    nbnew4=nb6(4)-nbcut4; nbnew5=0; nbnew6=0;
    
elseif Ab6(5)+Ab6(6)+Ab6(4)+Ab6(3)>Abtarget % rebar on the mid part of mid layer 
                                    % will be cut

    Abcut4=Abtarget-Ab6(5)-Ab6(6);
    nbcut4=fix(Abcut4/ab6(4));
    if mod(nb6(4),2)==0
        if mod(nbcut4,2)~=0
            nbcut4=nbcut4-1;
        end
    else
        if mod(nbcut4,2)==0 && nbcut4~=0
            nbcut4=nbcut4-1;
        end
    end
    Abcut=Ab6(5)+Ab6(6)+Ab6(4)+Ab6(3);

    nbnew1=nb6(1); nbnew2=nb6(2); nbnew3=0; 
    nbnew4=0; nbnew5=0; nbnew6=0;
    
else % rebars on the three layers will be cut
    Abcut2=Abtarget-Ab6(6)-Ab6(5)-Ab6(4)-Ab6(3);
    nbcut2=fix(Abcut2/ab6(2));
    if nb6(2)>=nbcut2
        if mod(nb6(2),2)==0
            if mod(nbcut2,2)~=0
                nbcut2=nbcut2-1;
            end
        else
            if mod(nbcut2,2)==0 && nbcut2~=0
                nbcut2=nbcut2-1;
            end
        end
        Abcut=Ab6(5)+Ab6(6)+Ab6(4)+Ab6(3)+nbcut2*ab6(2);
        
        nbnew1=nb6(1); nbnew2=nb6(2)-nbcut2; nbnew3=0; 
        nbnew4=0; nbnew5=0; nbnew6=0;
        
    else
        Abcut=Ab6(5)+Ab6(6)+Ab6(4)+Ab6(3)+Ab6(2);
        nbnew1=nb6(1); nbnew2=0; nbnew3=0; 
        nbnew4=0; nbnew5=0; nbnew6=0;
    end
    
end