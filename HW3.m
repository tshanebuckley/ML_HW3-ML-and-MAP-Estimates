% import coin.txt
data = importdata('coin.txt');
% knowing from class that the ML of a coin (H or T) is a bernouli dist with
% the final equation of Theta = #of_heads/total, calculate the ML from the
% data contained in  coin.txt
data = importdata('coin.txt');
s = size(data);
s = s(1);
heads = 0;
for c = 1:s
    if data(c) == 1
        heads = heads + 1;
    end
end
theta = heads/s
heads
tails = s - heads
% prior and posterior for Beta(Theta|1,1)
x = 0:0.01:1;
y1 = betapdf(x,1,1); % prior
y2 = betapdf(x,heads+1,tails+1); % posterior
% MAP estimate using prior Beta(Theta|1,1)
MAP_1_1 = (1+heads-1)/(1+1+s-2)
yMax_1_1 = max(y2);
% plot prior, posterior, and MAP for Beta(Theta|1,1)
figure()
plot([MAP_1_1 MAP_1_1],[0 yMax_1_1])
hold on 
plot(x,y1)
plot(x,y2)
title('Prior and Posterior for Beta(Theta|1,1)')
legend({'MAP','prior','posterior'},'Location','Northeast');
hold off
% prior and posterior for Beta(Theta|4,2)
x = 0:0.01:1;
y3 = betapdf(x,4,2); % prior
y4 = betapdf(x,heads+4,tails+2); % posterior
% MAP estimate using prior Beta(Theta|4,2)
MAP_4_2 = (4+heads-1)/(4+2+s-2)
yMax_4_2 = max(y4);
% plot prior, posterior, and MAP for Beta(Theta|4,2)
figure()
plot([MAP_4_2 MAP_4_2],[0 yMax_4_2])
hold on 
plot(x,y3)
plot(x,y4)
title('Prior and Posterior for Beta(Theta|4,2)')
legend({'MAP','prior','posterior'},'Location','Northeast');
hold off
% import gaussian.txt
data = importdata('gaussian.txt');
% graph gaussian.txt data
figure()
scatter(data(:,1),data(:,2))
hold on 
title('Data gaussian.txt')
hold off
% ML estimate of the mean for gaussian.txt
mean_gauss = mean(data)
cov_matrix_gauss = cov(data)
N = size(data);
N = N(1);
ML_mean = sum(data)/N
S = data - ML_mean;
S_T = transpose(S);
ML_cov_matrix = (S_T*S)/(N-1)
% plot 2D Gaussian using ML values
figure()
x1 = -5:.2:15; x2 = -5:.2:15;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],ML_mean,ML_cov_matrix);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
% split the data
m1 = data(:,1);
m2 = data(:,2);
% calculate ML mean and variance for m1
mean1 = mean(m1)
variance1 = var(m1)
ML_mean1 = sum(m1)/N
diff1 = m1-ML_mean1;
ML_variance1 = sum(diff1.^2)/(N-1)
% calculate ML mean and variance for m1
mean2 = mean(m2)
variance2 = var(m2)
ML_mean2 = sum(m2)/N
diff2 = m2-ML_mean2;
ML_variance2 = sum(diff2.^2)/(N-1)
% graph m1 and m2
x1 = -15:0.1:20;
x2 = -15:0.1:20;
norm1 = normpdf(x1,ML_mean1,ML_variance1);
norm2 = normpdf(x2,ML_mean2,ML_variance2);
figure()
plot(x1,norm1)
hold on
plot([ML_mean1 ML_mean1],[0 max(norm1)])
title('Data Column 1')
hold off
figure()
plot(x2,norm2)
hold on
plot([ML_mean2 ML_mean2],[0 max(norm2)])
title('Data Column 2')
hold off
% plot the exponential dist
x_exp = 0:0.1:10;
y_exp1 = exp(-x_exp);
y_exp2 = 4*exp((-x_exp)*4);
y_exp3 = exp((-x_exp)/4)/4;
figure()
plot(x_exp,y_exp1)
hold on
title('b = 1')
hold off
figure()
plot(x_exp,y_exp2)
hold on
title('b = 0.25')
hold off
figure()
plot(x_exp,y_exp3)
hold on
title('b = 4')
hold off