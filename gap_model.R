##-----model based on gap times (chapter 4)------------------------------------------------------------


# ------------  accelerated failure time (AFT) model --------------------------------------------------
# --------------------example (survreg) # 预测: survival.pdf (predict.survreg) --------------------------
head(lung)   # See: survival.pdf (predict.survreg)
lfit <- survreg(Surv(time, status) ~ ph.ecog, data=lung)
pct <- 1:98/100 # The 100th percentile of predicted survival is at +infinity
ptime <- predict(lfit, newdata=data.frame(ph.ecog=2), type='quantile', p=pct, se=TRUE)
matplot(cbind(ptime$fit, ptime$fit + 2*ptime$se.fit, ptime$fit - 2*ptime$se.fit)/30.5, 1-pct,
        xlab="Months", ylab="Survival", type='l', lty=c(1,2,2), col=1)
# ---------------------------------------------------

fit_surv <- survreg(Surv(time,event)~finish_order_num+complaint_7+complaint_30+coupon_7+
                     R_max+R_min+R_mean+F_max+F_min+F_mean+M_max+M_min+M_mean, data=rawdata_sp)
summary(fit_surv)
pre_cov <- rawdata_sp[1:50,6:19]
predict(fit_surv,newdata=pre_cov,se=TRUE)$fit

#
# =======================Question：====================================
# 1. 怎么fit recurrent event？ example是单一事件(每个人、是否发生)
# 2. 可以估计&预测，预测W还是log W ? 
# 

#---------------------------------- Cox model ------------------------------------------------
#------------ example: Create a simple data set for a time-dependent model------- 
test2 <- list(start=c(1,2,5,2,1,7,3,4,8,8), 
              stop=c(2,3,6,7,8,9,9,9,14,17), 
              event=c(1,1,1,1,1,1,1,0,0,0), 
              x=c(1,0,0,1,0,1,1,1,0,0)) 
summary(coxph(Surv(start, stop, event) ~ x, test2)) 

options(na.action=na.exclude) # retain NA in predictions
fit <- coxph(Surv(time, status) ~ age + ph.ecog + strata(inst), lung)
#lung data set has status coded as 1/2
mresid <- (lung$status-1) - predict(fit, type='expected') #Martingale resid
predict(fit,type="lp")
predict(fit,type="expected")

# ---------------------# 预测:基于 survival.pdf (predict.coxph)---------------
#-----example
options(na.action=na.exclude) # retain NA in predictions
fit <- coxph(Surv(time, status) ~ age + ph.ecog + strata(inst), lung)
#lung data set has status coded as 1/2
mresid <- (lung$status-1) - predict(fit, type='expected') #Martingale resid
predict(fit,type="lp")
predict(fit,type="expected")
#-------my dataset

#
# =======================Question：====================================
# 1. 怎么fit recurrent event？ example是单一事件(每个人、是否发生)
# 2. 可以估计&预测，预测的是什么 ?
# 

library (survival)


# Frailty models
model.frailty <- coxph(Surv(stop, event) ~ finish_order_num+complaint_7+complaint_30+coupon_7+
                        R_max+R_min+R_mean+F_max+F_min+F_mean+M_max+M_min+M_mean + frailty(ID), data=rawdata_sp)
summary(model.frailty)
pre <- predict(model.frailty,type='expected')
surv_prob <- exp(-pre)
event_prob <- 1-surv_prob

#Marginal means and rates model:
model.2 <- coxph(Surv(start,stop,event) ~ finish_order_num+complaint_7+complaint_30+coupon_7+
                   R_max+R_min+R_mean+F_max+F_min+F_mean+M_max+M_min+M_mean + cluster(ID), 
                   method="breslow", data = rawdata_sp)
summary(model.2)
pre <- predict(model.2,type='expected')
