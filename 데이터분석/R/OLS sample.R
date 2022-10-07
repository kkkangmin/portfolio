

# READ THE DATA FROM csv FILE 
dat <- read.csv("C:\\Users\\±è°­¹Î\\Desktop\\µ¥ºÐ±â ÇÁ·ÎÁ§Æ®\\lab2_data.csv", stringsAsFactors=F)
dat 

str(dat)
summary(dat)


newid<-dat$newid
hrsperwk<-dat$hrsperwk
marital<-dat$marital
sex<-dat$sex
age<-dat$age
wagex<-dat$wagex
asian<-dat$asian


# SCATTER PLOT 

plot(hrsperwk, wagex)


plot(age, wagex)


plot(sex, wagex) 



# USE least square model 

model1 <- lm(wagex~hrsperwk)
summary(model1) 

model2 <- lm(wagex~age) 
summary(model2)

model3 <- lm(wagex~hrsperwk+age) 
summary(model3)
anova(model1, model3) 

model4 <- lm(wagex~hrsperwk+age+sex) 
summary(model4)
anova(model1, model4) 
anova(model3, model4) 


# GENERATE NEW DATA 

#1 Replace null values with 0 for generating ASIAN-dummy 
asian[is.numeric(asian) & is.na(asian)] <- 0
summary(asian)
list(asian)

## Regression with Asian 
model5 <- lm(wagex~hrsperwk+age+sex+asian) 
summary(model5)

## F-Test on Asian dummy 
anova(model4, model5) 


#2 DUMMY*DUMMY INTERACTION
asian_sex <- asian*sex

## Regression with #2 interaction dummy 
model6 <- lm(wagex~hrsperwk+age+sex+asian+asian_sex) 
summary(model6)

#3 LOG TRANSFORMATION 
lnwagex = log(wagex) 
lnhrsperwk = log(hrsperwk)

plot(lnhrsperwk, lnwagex) 

modle16 <- lm(lnwagex~lnhrsperwk+age+sex+asian) 
summary(modle16)

#2 DUMMY*DUMMY INTERACTION
sex_marital <- sex*marital

## Regression with #2 interaction dummy 
model7 <- lm(wagex~hrsperwk+age+sex+asian+sex_marital) 
summary(model7)


#2 DUMMY*DUMMY INTERACTION
sex_hrsperwk <- sex*hrsperwk


## Regression with #2 interaction dummy 
model8 <- lm(wagex~hrsperwk+age+sex+asian+sex_hrsperwk) 
summary(model8)