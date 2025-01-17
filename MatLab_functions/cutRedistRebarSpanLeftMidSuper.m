function [nbnewl6,nbnewm6,nbnewr6,redistrRebarL2M,relistRebarDiamL2M,...
        redistrRebarM2L,relistRebarM2L,redistrRebarM2R,relistRebarM2R]=...
        cutRedistRebarSpanLeftMidSuper(load_conditions,nb6l,nb6m,db6l,db6m,...
        b,h,brec,hrec,hagg,fc,fy)

vsepl=max([2,max(db6l),hagg+0.5]);
Mleft=load_conditions(1,2);
Mmid=load_conditions(1,3);
Mright=load_conditions(1,4);
if Mleft*Mmid<0
    % Cuts Left-section
    
    Mtl=0.5*Mleft;
    [nbnew1l,nbnew2l,nbnew3l,nbnew4l,nbnew5l,nbnew6l,Abcutl]=cutRebarSecBeamMrSuper(nb6l,...
                                 db6l,Mtl,b,h,hrec,vsepl,fc,fy);

    % Re-distribution after cuts of rebar in tension
    % Mid section
    nbnewl6=[nbnew1l,nbnew2l,nbnew3l,nbnew4l,nbnew5l,nbnew6l];
    
    [redistrRebarL2M,relistRebarDiamL2M]=distrRebarRecBeamCutsSuper...
        (nb6l,db6l,b,h,brec,hrec,hagg,nbnewl6);
else
    nbnewl6=nb6l;
    [redistrRebarL2M,relistRebarDiamL2M]=distrRebarRecBeamCutsSuper...
        (nb6l,db6l,b,h,brec,hrec,hagg,nbnewl6);
end
vsepm=max([2,max(db6m),hagg+0.5]);
if Mmid*Mleft<0
    % Cuts mid-section
    Mtm=0.5*Mmid;
    [nbnew1m,nbnew2m,nbnew3m,nbnew4m,nbnew5m,nbnew6m,Abcutm]=cutRebarSecBeamMrSuper(nb6m,...
                                 db6m,Mtm,b,h,hrec,vsepm,fc,fy);
    % Re-distribution after cuts of rebar in tension
    % Left section
    nbnewm6=[nbnew1m,nbnew2m,nbnew3m,nbnew4m,nbnew5m,nbnew6m];
    [redistrRebarM2L,relistRebarM2L]=distrRebarRecBeamCutsSuper...
        (nb6m,db6m,b,h,brec,hrec,hagg,nbnewm6);
else
    if abs(Mleft)>0.5*abs(Mmid)
        nbnewm6=nb6m;

        [redistrRebarM2L,relistRebarM2L]=distrRebarRecBeamCutsSuper...
            (nb6m,db6m,b,h,brec,hrec,hagg,nbnewm6);
    elseif abs(Mleft)<=0.5*abs(Mmid)
        Mtm=0.5*Mmid;
        [nbnew1m,nbnew2m,nbnew3m,nbnew4m,nbnew5m,nbnew6m,Abcutm]=...
            cutRebarSecBeamMrSuper(nb6m,db6m,Mtm,b,h,hrec,vsepm,fc,fy);
        
        % Re-distribution after cuts of rebar in tension
        % Left section
        nbnewm6=[nbnew1m,nbnew2m,nbnew3m,nbnew4m,nbnew5m,nbnew6m];
        [redistrRebarM2L,relistRebarM2L]=distrRebarRecBeamCutsSuper...
            (nb6m,db6m,b,h,brec,hrec,hagg,nbnewm6);
    end
end
vsepr=vsepm;
% Cuts for right-section
if Mmid*Mright<0
    
    % Re-distribution after cuts of rebar in tension
    % Left section
    nbnewr6=nbnewm6;
    redistrRebarM2R=redistrRebarM2L;
    relistRebarM2R=relistRebarM2L;
else
    
    if abs(Mright)>0.5*abs(Mmid)
        nbnewr6=nb6m;

        [redistrRebarM2R,relistRebarM2R]=distrRebarRecBeamCutsSuper...
            (nb6m,db6m,b,h,brec,hrec,hagg,nbnewr6);
    elseif abs(Mright)<=0.5*abs(Mmid)
        Mtm=0.5*Mmid;
        [nbnew1r,nbnew2r,nbnew3r,nbnew4r,nbnew5r,nbnew6r,Abcutr]=cutRebarSecBeamMrSuper(nb6m,...
                                 db6m,Mtm,b,h,hrec,vsepr,fc,fy);
                             
        % Re-distribution after cuts of rebar in tension
        % Left section
        nbnewr6=[nbnew1r,nbnew2r,nbnew3r,nbnew4r,nbnew5r,nbnew6r];
        [redistrRebarM2R,relistRebarM2R]=distrRebarRecBeamCutsSuper...
            (nb6m,db6m,b,h,brec,hrec,hagg,nbnewr6);
    end
end