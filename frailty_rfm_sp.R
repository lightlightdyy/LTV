# 
# add stop and start variables into surv_300day.csv
#
data_surv <- read.csv('surv_data_300_sum_1m_sp.csv')[,-1]
data_surv['time']<- data_surv['stop']-data_surv['start']  #这里time变为gap time
data_surv <- data_surv[c(1,2,3,9,4,5,6,7,8)]
RMF <- read.csv('RMF_300_1m_sp.csv')[,-1]  #head(RMF)
colnames(RMF) <- c("R_max", "R_min", "R_mean", "F_max", "F_min", "F_mean", "M_max", 
                   "M_min", "M_mean", "R_range")
rawdata_sp <- cbind(data_surv,RMF)

dim(rawdata_sp) #24933    19
head(rawdata_sp) 
# summary(rawdata_sp)

cov_scale_sp <- apply(rawdata_sp,2,scale)[,6:19]
scale_data_sp <- cbind(rawdata_sp[,1:5],cov_scale_sp)
head(scale_data_sp)

#------------------------------------Event Counts with Random Effects -----------------------------------
#----------------------------frailtyPenal package (for Gamma frailty model)----------------------------
# example 1
frailtyPenal(Surv(time,status)~sex+age, n.knots=12,kappa=10000,data=kidney)

#example 2
head(readmission)
mod.cox.gap <- frailtyPenal(Surv(time, event) ~ chemo + sex + dukes + charlson, n.knots = 10, 
                            kappa = 1, data = readmission, cross.validation = TRUE)
summary(mod.cox.gap)
print(mod.cox.gap, digits = 4)
#----------------------------------------------------------------------------------------------

library("frailtypack")
fit_scale <- frailtyPenal(Surv(stop,event)~cluster(ID)+finish_order_num+complaint_7+complaint_30+coupon_7+
                      R_max+R_min+R_mean+F_max+F_min+F_mean+M_max+M_min+M_mean+R_range, 
                      n.knots=15, kappa=1000, data=scale_data_sp)
summary(fit_scale)
print(fit_scale,digits = 4)
# 剔除部分变量
fit_selection <- frailtyPenal(Surv(stop,event)~cluster(ID)+finish_order_num+complaint_30+coupon_7+
                          R_mean+F_mean+M_mean, n.knots=12,kappa=1000,data=scale_data_sp)
summary(fit_selection)
print(fit_selection, digits = 4)
pred <- prediction(fit_selection,scale_data_sp[1:100,c(1,3,5,6,8,9,10,11,12,15,18)],t=30,window=seq(1,60,1),
                   event='Recurrent',conditional = TRUE)
# predictive probability of event between t and horizon time t+w, conditional：given a specific individual

#
# ======================== Question ==============================
# 可以直接fit recurrent event data (gamma frailty model)
# 可以用prediction 预测，预测[t,t+w]内事件发生的概率
# 预测集形式不确定
# example 中covariate是time-fixed,且为factor变量


