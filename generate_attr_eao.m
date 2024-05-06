%%%%%   MDNet   %%%
Ours.all = 0.362;

%%%%%   MDNet   %%%
SiamMask.all = 0.342;

%%%%%   MDNet   %%%
SiamCAR.all = 0.211;

%%%%%   SiamFC   %%%
SiamRPNpp.all = 0.166;

%%%%%   SiamFC   %%%
SiamRPN.all = 0.143;











result.Ours = Ours;
result.SiamMask = SiamMask;

result.SiamCAR = SiamCAR;
result.SiamRPNpp = SiamRPNpp;
result.SiamRPN = SiamRPN;




save(['attr_eao_vot2016.mat'], 'result'); 
