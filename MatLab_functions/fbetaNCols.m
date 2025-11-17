function fbeta=fbetaNCols(pu,fcu,b,h)

    Nbhfcu=abs(pu)/(b*h*fcu);
    if all([Nbhfcu>=0,Nbhfcu<0.1])
        fbeta=1-0.12/0.1*Nbhfcu;
    elseif all([Nbhfcu>=0.1,Nbhfcu<0.2])
        fbeta=0.88-0.11/0.1*(Nbhfcu-0.1);
    elseif all([Nbhfcu>=0.2,Nbhfcu<0.3])
        fbeta=0.77-0.12/0.1*(Nbhfcu-0.2);
    elseif all([Nbhfcu>=0.3,Nbhfcu<0.4])
        fbeta=0.65-0.12/0.1*(Nbhfcu-0.3);
    elseif all([Nbhfcu>=0.4,Nbhfcu<0.5])
        fbeta=0.53-0.11/0.1*(Nbhfcu-0.4);
    elseif all([Nbhfcu>=0.5,Nbhfcu<0.6])
        fbeta=0.42-0.12/0.1*(Nbhfcu-0.5);
    elseif Nbhfcu>=0.6
        fbeta=0.3;
    end
end