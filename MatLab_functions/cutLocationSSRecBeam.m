function cutxLoc=cutLocationSSRecBeam(M,dL)

ne=length(M(1,:));
Mleft=M(1,1);
Mright=M(1,ne);
[Mmid,mp]=max(M(1,:));

%% Cut location
if Mleft*Mmid<0 && Mright*Mmid<0 % Standard case
    cutxLoc=[];
    for n=1:ne
        Mref=Mleft;
        if abs(M(2,n))<=abs(0.5*Mref)
            n1=n;
            cutxLoc=[cutxLoc,dL*n];
            break;
        end
    end

    Mref=Mmid;
    for n=mp:-1:1
        if abs(M(2,n))<=abs(0.5*Mref)
            cutxLoc=[cutxLoc,dL*n];
            break;
        end
    end

    Mref=Mmid;
    for n=mp:1:ne
        if abs(M(2,n))<=abs(0.5*Mref)
            cutxLoc=[cutxLoc,dL*n];
            break;
        end
    end

    Mref=Mright;
    for n=ne:-1:1
        if abs(M(2,n))<=abs(0.5*Mref)
            cutxLoc=[cutxLoc,dL*n];
            break;
        end
    end
elseif Mleft*Mmid<0 && Mright*Mmid>0
    if abs(Mmid)*0.5>abs(Mright) % Case 2.1.a
        cutxLoc=[];
        for n=1:ne
            Mref=Mleft;
            if abs(M(2,n))<=abs(0.5*Mref)
                n1=n;
                cutxLoc=[cutxLoc,dL*n];
                break;
            end
        end

        Mref=Mmid;
        for n=mp:-1:1
            if abs(M(2,n))<=abs(0.5*Mref)
                cutxLoc=[cutxLoc,dL*n];
                break;
            end
        end
        Mref=Mmid;
        for n=mp:1:ne
            if abs(M(2,n))<=abs(0.5*Mref)
                cutxLoc=[cutxLoc,dL*n];
                break;
            end
        end
        
        cutxLoc=[cutxLoc,dL*ne];
    elseif abs(Mmid)*0.5<=abs(Mright) % Case 2.2.a
        cutxLoc=[];
        for n=1:ne
            Mref=Mleft;
            if abs(M(2,n))<=abs(0.5*Mref)
                n1=n;
                cutxLoc=[cutxLoc,dL*n];
                break;
            end
        end

        Mref=Mmid;
        for n=mp:-1:1
            if abs(M(2,n))<=abs(0.5*Mref)
                cutxLoc=[cutxLoc,dL*n];
                break;
            end
        end
        cutxLoc=[cutxLoc,dL*ne];
        cutxLoc=[cutxLoc,dL*ne];
    end
elseif Mleft*Mmid>0 && Mright*Mmid<0
    if abs(Mmid)*0.5>abs(Mleft) % Case 2.1.b
        cutxLoc=[];
        cutxLoc=[cutxLoc,0];
        Mref=Mmid;
        for n=mp:-1:1
            if abs(M(2,n))<=abs(0.5*Mref)
                cutxLoc=[cutxLoc,dL*n];
                break;
            end
        end
        
        Mref=Mmid;
        for n=mp:1:ne
            if abs(M(2,n))<=abs(0.5*Mref)
                cutxLoc=[cutxLoc,dL*n];
                break;
            end
        end
        
        
        Mref=Mright;
        for n=ne:-1:1
            if abs(M(2,n))<=abs(0.5*Mref)
                cutxLoc=[cutxLoc,dL*n];
                break;
            end
        end
        
    elseif abs(Mmid)*0.5<=abs(Mleft) % Case 2.2.b
        cutxLoc=[]; 
        cutxLoc=[cutxLoc,0];
        cutxLoc=[cutxLoc,0];
        
        Mref=Mmid;
        for n=mp:1:ne
            if abs(M(2,n))<=abs(0.5*Mref)
                cutxLoc=[cutxLoc,dL*n];
                break;
            end
        end
        
        
        Mref=Mright;
        for n=ne:-1:1
            if abs(M(2,n))<=abs(0.5*Mref)
                cutxLoc=[cutxLoc,dL*n];
                break;
            end
        end
    end
    
end