
function Iatt = test_awgn(Iw, noisePower, seed)

rand('state',seed);
 % Seed random generator

Iatt = imnoise(Iw,'gaussian',0,noisePower);

