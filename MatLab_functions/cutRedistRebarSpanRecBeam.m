function [nbnewl6,nbnewm6,nbnewr6,redistrRebarL2M,relistRebarDiamL2M,...
        redistrRebarM2L,relistRebarM2L,redistrRebarM2R,relistRebarM2R]=...
        cutRedistRebarSpanRecBeam(load_conditions,nb6l,nb6m,db6l,db6m,...
        b,h,brec,hrec,hagg,fc,fy)
    
    %------------------------------------------------------------------------
    % Syntax:
    % [nbnewl6,nbnewm6,nbnewr6,redistrRebarL2M,relistRebarDiamL2M,...
    %   redistrRebarM2L,relistRebarM2L,redistrRebarM2R,relistRebarM2R]=...
    %   cutRedistRebarSpanRecBeam(load_conditions,nb6l,nb6m,db6l,db6m,...
    %   b,h,brec,hrec,hagg,fc,fy)
    %
    %-------------------------------------------------------------------------
    % SYSTEM OF UNITS: Any.
    %
    %------------------------------------------------------------------------
    % PURPOSE: To compute the redistribution of cross-section rebar design 
    % coordinates, according to a original rebar distribution and the number
    % cuts to execute. The rebar design must consist of a max of 12 diameter
    % sizes, along the whole length of the beam.
    % 
    % OUTPUT: redistrRebarM2R:    vector (nbars x 2) that contains the new
    %                             rebar cross-section coordinates on the 
    %                             right beam cross-section from the mid
    %                             cross-section, after rebar cutting
    %
    % INPUT:  nb6l:               is the vector containing the number of 
    %                             rebars for each rebar layer component 
    %                             (1 x 6) for the left cross-section of a 
    %                             beam
    %
    %------------------------------------------------------------------------
    % LAST MODIFIED: L.F.Veduzco    2025-02-05
    %                School of Engineering
    %                The Hong Kong University of Science and Technology (HKUST)
    %------------------------------------------------------------------------
    
    [vsepl,~]=sepMinMaxHK13(db6l,hagg,1);
    Mleft=load_conditions(1,2);
    Mmid=load_conditions(1,3);
    Mright=load_conditions(1,4);
    if Mleft*Mmid<0
        % Cuts Left-section
        
        Mtl=0.5*Mleft;
        [nbnew1l,nbnew2l,nbnew3l,nbnew4l,nbnew5l,nbnew6l,Abcutl]=cutRebarSecBeamMr...
            (nb6l,db6l,Mtl,b,h,hrec,vsepl,fc,fy);
    
        % Re-distribution after cuts of rebar in tension
        % Mid section
        nbnewl6=[nbnew1l,nbnew2l,nbnew3l,nbnew4l,nbnew5l,nbnew6l];
        
        [redistrRebarL2M,relistRebarDiamL2M]=distrRebarRecBeamCuts...
            (nb6l,db6l,b,h,brec,hrec,vsepl,nbnewl6);
    else
        nbnewl6=nb6l;
        [redistrRebarL2M,relistRebarDiamL2M]=distrRebarRecBeamCuts...
            (nb6l,db6l,b,h,brec,hrec,vsepl,nbnewl6);
    end
    
    [vsepm,vsepmaxm]=sepMinMaxHK13(db6m,hagg,1);
    if Mmid*Mleft<0
        % Cuts mid-section
        Mtm=0.5*Mmid;
        [nbnew1m,nbnew2m,nbnew3m,nbnew4m,nbnew5m,nbnew6m,Abcutm]=cutRebarSecBeamMr(nb6m,...
                                     db6m,Mtm,b,h,hrec,vsepm,fc,fy);
        % Re-distribution after cuts of rebar in tension
        % Left section
        nbnewm6=[nbnew1m,nbnew2m,nbnew3m,nbnew4m,nbnew5m,nbnew6m];
        [redistrRebarM2L,relistRebarM2L]=distrRebarRecBeamCuts...
            (nb6m,db6m,b,h,brec,hrec,vsepm,nbnewm6);
    else
        if abs(Mleft)>0.5*abs(Mmid)
            nbnewm6=nb6m;
    
            [redistrRebarM2L,relistRebarM2L]=distrRebarRecBeamCuts...
                (nb6m,db6m,b,h,brec,hrec,vsepm,nbnewm6);
        elseif abs(Mleft)<=0.5*abs(Mmid)
            Mtm=0.5*Mmid;
            [nbnew1m,nbnew2m,nbnew3m,nbnew4m,nbnew5m,nbnew6m,Abcutm]=...
                cutRebarSecBeamMr(nb6m,db6m,Mtm,b,h,hrec,vsepm,fc,fy);
            
            % Re-distribution after cuts of rebar in tension
            % Left section
            nbnewm6=[nbnew1m,nbnew2m,nbnew3m,nbnew4m,nbnew5m,nbnew6m];
            [redistrRebarM2L,relistRebarM2L]=distrRebarRecBeamCuts...
                (nb6m,db6m,b,h,brec,hrec,vsepm,nbnewm6);
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
    
            [redistrRebarM2R,relistRebarM2R]=distrRebarRecBeamCuts...
                (nb6m,db6m,b,h,brec,hrec,vsepr,nbnewr6);
        elseif abs(Mright)<=0.5*abs(Mmid)
            Mtm=0.5*Mmid;
            [nbnew1r,nbnew2r,nbnew3r,nbnew4r,nbnew5r,nbnew6r,Abcutr]=cutRebarSecBeamMr(nb6m,...
                                     db6m,Mtm,b,h,hrec,vsepr,fc,fy);
                                 
            % Re-distribution after cuts of rebar in tension
            % Left section
            nbnewr6=[nbnew1r,nbnew2r,nbnew3r,nbnew4r,nbnew5r,nbnew6r];
            [redistrRebarM2R,relistRebarM2R]=distrRebarRecBeamCuts...
                (nb6m,db6m,b,h,brec,hrec,vsepr,nbnewr6);
        end
    end
end