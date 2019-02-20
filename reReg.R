library(reda)

#tmp <- read.csv('surv_data_500_freq.csv')#[,-c(1,7,8)]
tmp <- read.csv('surv_data_300_train_2.csv')[,-1]
head(tmp,20)

RMF <- read.csv('rfm_feature_300_train.csv',sep=",")[,-1]   #RMF <- read.csv('rfm_feature.tsv',sep="\t")   
colnames(RMF) <- c("R_max", "R_min", "R_mean", "F_max", "F_min", "F_mean", "M_max", "M_min", "M_mean", "R_range")
count_ID <- as.data.frame(table(tmp[,1]))[,2]
da <- data.frame(matrix(NA,dim(tmp)[1],10));colnames(da) <- colnames(RMF)

for(i in 1:10){
  da[,i] <- rep(RMF[,i],count_ID)
  } #将RMF处理成event列表形式

covariate <- (cbind(tmp,da)[,-c(1,2,3)]) # head(covariate) # summary(covariate)
cov_scale <- apply(covariate,2,scale)
data <- cbind(tmp[,1:3],cov_scale[,c(1:4,6:7,9,10,13)])
head(data)
rawdata <- cbind(tmp[,1:3],cov_scale)
head(rawdata)
#1.数据standalize
#2.现有统计方法为什么不好？
#3.对哪一类人不work？

## constant rate function
constFit <- rateReg(Survr(ID, time, event) ~ R_max+R_mean+F_max+F_mean+M_max+R_range, data) 
summary(constFit)
constFit1 <- rateReg(Survr(ID, time, event) ~ finish_order_num + complaint + coupon + R_max + 
                    R_mean + F_max + F_mean + M_max + R_range,data,dist='exponential')
summary(constFit1)  #加入GMV后全部变量不显著了。。

## six pieces' piecewise constant rate function
piecesFit <- rateReg(Survr(ID,time,event)~ finish_order_num+complaint+coupon+R_max
                    +R_mean+F_max+F_mean+M_max+R_range, data,knots = seq.int(0, 300, by = 50))
summary(piecesFit)
## fit rate function with cubic spline
splineFit <- rateReg(Survr(ID, time, event) ~ finish_order_num+complaint+coupon+R_max+
                       R_mean+F_max+F_mean+M_max+R_range, data,knots = c(100, 200, 280), degree = 3)
summary(splineFit)

## model selection based on AIC or BIC
AIC(constFit, piecesFit, splineFit)
BIC(constFit, piecesFit, splineFit)

## estimated covariate coefficients
coef(piecesFit)
coef(splineFit)

## confidence intervals for covariate coefficients
confint(piecesFit)
confint(splineFit, "M_max", 0.9)
confint(splineFit, 1, 0.975)

## estimated baseline rate function
constbase <- baseRate(constFit)
plot(constbase,conf.int=TRUE)

piecesbase <- baseRate(piecesFit)
plot(piecesbase,conf.int=TRUE)

splinesBase <- baseRate(splineFit)
plot(splinesBase, conf.int = TRUE)


## estimated baseline mean cumulative function (MCF) from a fitted model
constMcf <- mcf(piecesFit)
plot(constMcf, conf.int = TRUE, col = "blueviolet")

piecesMcf <- mcf(piecesFit)
plot(piecesMcf, conf.int = TRUE, col = "blueviolet")

integrate(function(t){1+t},lower = 0, upper = 1)$value #$abs.error



test <- read.csv('surv_data_test_65.csv')[,-1]  #每人后65天的每天打车次数

## estimated MCF for given new data
newDat <- data.frame(x1 = rep(0, 2), group = c("Treat", "Contr"))
splineMcf <- mcf(splineFit, newdata = newDat, groupName = "Group",
                 groupLevels = c("Treatment", "Control"))
plot(splineMcf, conf.int = TRUE, lty = c(1, 5))

## example of further customization by ggplot2
library(ggplot2)
plot(splineMcf) +
  geom_ribbon(aes(x = time, ymin = lower,
                  ymax = upper, fill = Group),
              data = splineMcf@MCF, alpha = 0.2) +
  xlab("Days")











# reReg package
data(readmission, package = "frailtypack")   # ??readmission
set.seed(123)
## Accelerated Mean Model
fit <- reReg(Survr(ID,time,event) ~ coupon,data=tmp,
              method = "am.GL", se = "resampling", B = 20)
summary(fit)

fit <- reReg(reSurv(t.stop, id, event, death) ~ sex + chemo,
             data = subset(readmission, id < 50),
             method = "am.GL", se = "resampling", B = 20)






library(stats)
# NOT RUN { 
require(graphics) 
with(cars, { 
plot(speed, dist) 
lines(ksmooth(speed, dist, "normal", bandwidth = 2), col = 2) 
lines(ksmooth(speed, dist, "normal", bandwidth = 5), col = 3) 
}) # }
