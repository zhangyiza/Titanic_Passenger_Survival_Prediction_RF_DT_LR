#Apriori
train<-read.csv("train_3.csv")
titl<-read.csv("title.csv",sep = "\t")
title<-titl$Title
test<-read.csv("train_try.csv")
library(car)
library(boot)
#全部变量，不含交叉项
logit0<-glm(formula = Survived~Sex+Pclass+Age+familysize+averageFare+title+Embarked,
family = binomial(),data = train)
summary(logit0)
cv0<-cv.glm(train,logit0,K=6)
cv0$delta
vif(logit0)
#去掉 title 和 Embarked
logit1<-glm(formula = Survived~Sex+Pclass+Age+familysize+averageFare, family =
binomial(),data = train)
summary(logit1)
cv1<-cv.glm(train,logit1,K=6)
cv1$delta
vif(logit1)
#Age 分老人小孩
logit2<-glm(formula = Survived~Sex+Pclass+Age2+familysize+averageFare, family =
binomial(),data = train)
summary(logit2)
cv2<-cv.glm(train,logit2,K=6)
cv2$delta
vif(logit2)
#不考虑是不是老年人， Pclass 离散化
logit3<-glm(formula = Survived~Sex+Pclass2+Age3+familysize+averageFare, family =
binomial(),data = train)
summary(logit3)
cv6<-cv.glm(train,logit3,K=6)
cv6$delta
vif(logit3)
#加入交互项
logit.inter3<-glm(formula =
Survived~Sex+Pclass2+familysize+averageFare+Age3*Sex+Sex*familysize+Pclass2*Sex, family
= binomial(),data = train)
summary(logit.inter3)
cv5<-cv.glm(train,logit.inter3,K=6)
cv5$delta
vif(logit.inter3)
#去掉 Pclass2*Age3
logit.inter4<-glm(formula =
Survived~Sex+Pclass2+familysize+averageFare+Age3*Sex+Sex*familysize, family =
binomial(),data = train)
summary(logit.inter4)
cv7<-cv.glm(train,logit.inter4,K=6)
cv7$delta
vif(logit.inter4)
#Apriori
titanic<-read.csv("AAnalysis12.csv")
# attach packages
library(arules)
library(arulesViz)
library(ggplot2)
library(plyr)
rule<-apriori(titanic)
rule
inspect(rule)
plot(rule,method="graph")
rule1 <- apriori(titanic,
parameter=list(minlen=1, supp=0.005, conf=0.8),
appearance=list(rhs=c("Survived=no", "Survived=yes"),
default="lhs") )
inspect(rule1)
plot(rule1,method="graph")
# find redundant rules
matrix <- is.subset(rule1, rule1) # check whether one set is a subset of others
matrix[ lower.tri(matrix, diag=T) ] <- FALSE # set all diag values be FALSE
Redundant.rowNum <- colSums(matrix, na.rm=T) >= 1 # find out all the redundant rules
rule1.new <- rule1[!Redundant.rowNum] # remove all the redundant rules
inspect(rule1.new) # inspect our new set of rules
plot(rule1.new,method="graph")
rule.embarked<-apriori(titanic,
parameter=list(minlen=1, supp=0.01, conf=0.8),
appearance=list(lhs=c("Embarked=S", "Embarked=C","Embarked=Q"),
default="rhs") )
inspect(rule.embarked)