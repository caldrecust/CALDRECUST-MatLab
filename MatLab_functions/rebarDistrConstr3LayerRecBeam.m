function [cc]=rebarDistrConstr3LayerRecBeam(bp,nb6l)

nbl1=nb6l(1);
nbl2=nb6l(2);
nbl3=nb6l(3);
nbl4=nb6l(4);
nbl5=nb6l(5);
nbl6=nb6l(6);

%% Checking restrictions of rebar distribution 1
% Note, there cannot be only one rebar at the ends of each layer.
% According to code standards, there must be two rebars, to
% facilitate link placement.

% Left & mid section
if any([nbl3==1,nbl5==1])
    solbarL0=0;
else
    solbarL0=1;
end

%% Checking restrictions of rebar distribution 2

% Notes: a value of solbarL2 = 1 means that the restrictions 
% of distribution comply. These restrictions have to do with rebar
% ALIGNMENT to facilitate the placement of shear links.

if nbl5>nbl1 || nbl3>nbl1
    solbarL1=0;
else
    solbarL1=1;
end

sepf1L=(bp)/(nbl1+nbl2-1);

if nbl3+nbl4>1 % if there is a second layer of rebar
    sepf2L=(bp)/(nbl3+nbl4-1);
    if mod(sepf2L,sepf1L)~=0 % to ensure alignment with the first layer
        solbarL2=0;
    else
        solbarL2=1; % OK
    end
    
elseif nbl3+nbl4==1
    if mod(nbl1+nbl2,2)==0 % if the number of rebar in the first layer is 
                      % pair
        solbarL2=0;
    else
        solbarL2=1; % OK
    end
else
    solbarL2=1;
end

if nbl5+nbl6>1
    sepf3L=(bp)/(nbl5+nbl6-1);
    if mod(sepf3L,sepf1L)~=0
        solbarL3=0;
    else
        solbarL3=1; % OK
    end
elseif nbl5+nbl6==1
    if mod(nbl1+nbl2,2)==0
        solbarL3=0;
    else
        solbarL3=1; % OK
    end
else
    solbarL3=1;
end

%% Set of restriction of rebar distribution 3
% The number of rebar at the bottom layer (1st layer) must always be larger
% than the number of rebar at the upper layers, to ensure a proper
% placement of shear links
if all([nbl2 >= nbl4, nbl2 >= nbl6])
    solbarL4=1;
else
    solbarL4=0;
end

% To ensure symmetry of placement with respect to the vertical axis
% and alignment of rebars on top of bottom layer
if mod(nbl2,2)==0 % if the number of rebars on the bottom layer is pair 
    if any([mod(nbl4,2)==1,mod(nbl6,2)==1]) 
        solbarL5=0; % then the number of rebars on the top layers cannot
    else            % be impair
        solbarL5=1;
    end
else
    solbarL5=1;
end

if mod(nbl4,2)==0 
    if mod(nbl6,2)==1
        solbarL6=0;
    else
        solbarL6=1;
    end
else
    solbarL6=1;
end

%% Checking all restriction compliance
if solbarL0==1 && ...
   solbarL1==1 && ... 
   solbarL2==1 && ... 
   solbarL3==1 && ...
   solbarL4==1 && ...
   solbarL5==1 && ...
   solbarL6==1
    
    cc=1;
else
    cc=0;
end
