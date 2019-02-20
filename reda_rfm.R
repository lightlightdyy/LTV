data_surv <- read.csv('surv_data_300_sum_1m.csv')[,-1]
RMF <- read.csv('RMF_300_1m.csv')[,-1]  #head(RMF)
colnames(RMF) <- c("R_max", "R_min", "R_mean", "F_max", "F_min", "F_mean", "M_max", 
                   "M_min", "M_mean", "R_range")
rawdata <- cbind(data_surv,RMF)
dim(rawdata) #22000*17

cov_scale <- apply(rawdata,2,scale)[,4:17]
scale_data <- cbind(rawdata[,1:3],cov_scale)
head(scale_data)
summary(scale_data)


# ---------------- reda package.pdf  (for gamma frality model) -------------

const <- rateReg(Survr(ID, time, event) ~ finish_order_num+complaint_7+complaint_30+coupon_7+
                    R_max+R_min+R_mean+F_max+F_min+F_mean+M_max+M_min+M_mean, data=rawdata)
summary(const)  #用scale_data 奇异, 无解

#
# ================ Question========================
#
# 可以直接fit recurrent event data, 也可以估计出参数(gamma frailty model)
# 没有现成的预测： predict(const,head(rawdata))  报错："predict"没有适用于"rateReg"目标对象的方法
# 自己编写函数预测太麻烦， baseline function











splineFit <- rateReg(Survr(ID, time, event) ~ finish_order_num+complaint_7+complaint_30+coupon_7+
                    R_max+R_min+R_mean+F_max+F_min+F_mean+M_max+M_min+M_mean+R_range, 
                    data=scale_data, knots = c(100, 180, 250), degree = 3) #knots = seq.int(0, 298, by = 50)
summary(splineFit)

## model selection based on AIC or BIC
AIC(const, piecesFit, splineFit)  # BIC(constFit, piecesFit, splineFit)

## estimated baseline rate function
conBase <- baseRate(const)
plot(conBase, conf.int = TRUE)

## estimated baseline mean cumulative function (MCF) from a fitted model
constMcf <- mcf(const)
plot(constMcf, conf.int = TRUE, col = "blueviolet")


data(readmission, package = "frailtypack")
set.seed(123)
## Accelerated Mean Model
fit <- reReg(reSurv(t.stop, id, event, death) ~ sex + chemo,data = subset(readmission, id < 50),
              method = "am.XCHWY", se = "resampling", B = 20)
summary(fit)



## ------For rateReg object, mcf estimates the 【baseline MCF】 for given new data--------
newDat <- data.frame(x1 = rep(0, 2), group = c("Treat", "Contr"))
splineMcf <- mcf(splineFit, newdata = newDat, groupName = "Group", groupLevels = c("Treatment", "Control"))
plot(splineMcf, conf.int = TRUE, lty = c(1, 5))
## example of further customization by ggplot2
library(ggplot2)
plot(splineMcf) +
  geom_ribbon(aes(x = time, ymin = lower,ymax = upper, fill = Group),
              data = splineMcf@MCF, alpha = 0.2) + xlab("Days")


