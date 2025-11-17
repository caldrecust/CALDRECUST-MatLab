function [nbnewl3,nbnewm3,nbnewr3,redistrRebarL2M,relistRebarDiamL2M,...
        redistrRebarM2L,relistRebarM2L,redistrRebarM2R,relistRebarM2R]=...
        cutRedistRebarSpanRecBeam1DSec(load_conditions,nb3l,nb3m,dbl,dbm,...
        b,h,brec,hrec,hagg,fc,fy)

[vsepl,~]=sepMinMaxHK13(dbl,hagg,1);
Mleft=load_conditions(1,2);
Mmid=load_conditions(1,3);
Mright=load_conditions(1,4);
if Mleft*Mmid<0
    % Cuts Left-section
    
    Mtl=0.5*Mleft;
    [nbnew1l,nbnew2l,nbnew3l,Abcutl]=cutRebarSecBeamMr1DSec...
        (nb3l,dbl,Mtl,b,h,hrec,vsepl,fc,fy);

    % Re-distribution after cuts of rebar in tension
    % Mid section
    nbnewl3=[nbnew1l,nbnew2l,nbnew3l];
    
    [redistrRebarL2M,relistRebarDiamL2M]=distrRebarRecBeamCuts1DSec...
        (nb3l,dbl,b,h,brec,hrec,vsepl,nbnewl3);
else
    nbnewl3=nb3l;
    [redistrRebarL2M,relistRebarDiamL2M]=distrRebarRecBeamCuts1DSec...
        (nb3l,dbl,b,h,brec,hrec,vsepl,nbnewl3);
end

[vsepm,vsepmaxm]=sepMinMaxHK13(dbm,hagg,1);
if Mmid*Mleft<0
    % Cuts mid-section
    Mtm=0.5*Mmid;
    [nbnew1m,nbnew2m,nbnew3m,Abcutm]=cutRebarSecBeamMr1DSec(nb3m,...
                                 dbm,Mtm,b,h,hrec,vsepm,fc,fy);
    % Re-distribution after cuts of rebar in tension
    % Left section
    nbnewm3=[nbnew1m,nbnew2m,nbnew3m];
    [redistrRebarM2L,relistRebarM2L]=distrRebarRecBeamCuts1DSec...
        (nb3m,dbm,b,h,brec,hrec,vsepm,nbnewm3);
else
    if abs(Mleft)>0.5*abs(Mmid)
        nbnewm3=nb3m;

        [redistrRebarM2L,relistRebarM2L]=distrRebarRecBeamCuts1DSec...
            (nb3m,dbm,b,h,brec,hrec,vsepm,nbnewm3);
    elseif abs(Mleft)<=0.5*abs(Mmid)
        Mtm=0.5*Mmid;
        [nbnew1m,nbnew2m,nbnew3m,Abcutm]=...
            cutRebarSecBeamMr1DSec(nb3m,dbm,Mtm,b,h,hrec,vsepm,fc,fy);
        
        % Re-distribution after cuts of rebar in tension
        % Left section
        nbnewm3=[nbnew1m,nbnew2m,nbnew3m];
        [redistrRebarM2L,relistRebarM2L]=distrRebarRecBeamCuts1DSec...
            (nb3m,dbm,b,h,brec,hrec,vsepm,nbnewm3);
    end
end
vsepr=vsepm;
% Cuts for right-section
if Mmid*Mright<0
    
    % Re-distribution after cuts of rebar in tension
    % Left section
    nbnewr3=nbnewm3;
    redistrRebarM2R=redistrRebarM2L;
    relistRebarM2R=relistRebarM2L;
else
    
    if abs(Mright)>0.5*abs(Mmid)
        nbnewr3=nb3m;

        [redistrRebarM2R,relistRebarM2R]=distrRebarRecBeamCuts1DSec...
            (nb3m,dbm,b,h,brec,hrec,vsepr,nbnewr3);
    elseif abs(Mright)<=0.5*abs(Mmid)
        Mtm=0.5*Mmid;
        [nbnew1r,nbnew2r,nbnew3r,Abcutr]=cutRebarSecBeamMr1DSec(nb3m,...
                                 dbm,Mtm,b,h,hrec,vsepr,fc,fy);
                             
        % Re-distribution after cuts of rebar in tension
        % Left section
        nbnewr3=[nbnew1r,nbnew2r,nbnew3r];
        [redistrRebarM2R,relistRebarM2R]=distrRebarRecBeamCuts1DSec...
            (nb3m,dbm,b,h,brec,hrec,vsepr,nbnewr3);
    end
end