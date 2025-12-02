function [nbnew1,nbnew2,nbnew3,nbnew4,nbnew5,nbnew6,Abcut]=...
                cutRebarSecBeamMr6DSec(nb6,db6,Mt,b,h,hrec,vSep,fcu,fy)
   
    %------------------------------------------------------------------------
    % Syntax:
    % [nbnew1,nbnew2,nbnew3,nbnew4,nbnew5,nbnew6,Abcut]=...
    %            cutRebarSecBeamMr6DSec(nb6,db6,Mt,b,h,hrec,vSep,fcu,fy)
    %
    %-------------------------------------------------------------------------
    % SYSTEM OF UNITS: Any.
    %
    %------------------------------------------------------------------------
    % PURPOSE: To compute the amount and number of rebars to be cut for a
    % beam cross-section according to a target bending moment (less or
    % equal than the one with which the original rebar design was carried
    % out). The rebar design must consist of a max of 6 diameter sizes.
    % 
    % OUTPUT: Abcut:    total rebar cross-section are to be cut
    %
    % INPUT:  nb6:      is the vector containing the number of rebars for
    %                   each rebar layer component (1 x 6) for a beam
    %                   cross-section
    %
    %------------------------------------------------------------------------
    % LAST MODIFIED: L.F.Veduzco    2025-02-05
    %                School of Engineering
    %                The Hong Kong University of Science and Technology (HKUST)
    %------------------------------------------------------------------------
            
    Amin=0.003*b*h;
    
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
    
    k=abs(Mt/(b*d^2*fcu));
    if k>0.156
        k=0.156;
    end
    z=d*(0.5+sqrt(0.25-k/0.9));
    Abm=Mt/(0.87*fy*z);
    Abm=max([Abm,Amin]);
    
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
end