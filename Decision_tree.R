#Decision Tree
data <- read.table(file = "./data/train.csv", header = TRUE, sep = ",")
#classification of names
data$Name <- as.character(data$Name)
strsplit(data$Name[1], split='[,.]')[[1]][2]
data$Title <- sapply(data$Name, FUN = function(x){strsplit(x, split = '[,.]')[[1]][2]})
data$Title <- sub(' ','',data$Title)
table(data$Title)
data$Title[data$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
data$Title[data$Title %in% c('Capt', 'Don', 'Major', 'Sir', 'Jonkheer')] <- 'Sir'
data$Title[data$Title %in% c('Dona', 'Lady', 'the Countess')] <- 'Lady'
data$Title <- factor(data$Title)
#family size
data$FamilySize <- data$SibSp + data$Parch + 1
data$Surname <- sapply(data$Name, FUN = function(x){strsplit(x, split =
'[,.]')[[1]][1]})
data$FamilyID <- paste(as.character(data$FamilySize), data$Surname, sep = "")
data$FamilyID[data$FamilySize <= 2] <- "Small"
table(data$FamilyID)
famIDs <- data.frame(table(data$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2, ]
data$FamilyID[data$FamilyID %in% famIDs$Var1] <- 'Small'
data$FamilyID <- factor(data$FamilyID)
#model1
#no other variables
library(rpart)
library(rpart.plot)
fit1 <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
data=data, method="class")
printcp(fit1)
fit11 <- prune(fit1, cp= fit1$cptable[which.min(fit1$cptable[,"xerror"]),"CP"])
rpart.plot(fit11, branch=1, branch.type=2, type=2, extra=102,
shadow.col="gray", box.col="darkseagreen3",
border.col="darkseagreen3", split.col="chocolate4",
split.cex=1.2, main="Decision Tree of Titanic Survivals", uniform=T)
#model3
#add title and family size
fit3 <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title +
FamilySize, data=data, method="class")
printcp(fit3)
fit33 <- prune(fit3, cp= fit3$cptable[which.min(fit3$cptable[,"xerror"]),"CP"])
summary(fit33)
rpart.plot(fit33, branch=1, branch.type=2, type=2, extra=102,
shadow.col="gray", box.col="darkseagreen3",
border.col="darkseagreen3", split.col="chocolate4",
split.cex=1.2, main="Decision Tree of Titanic Survivals", uniform=T)
#model5
#all interactions
fit5 <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked +
Title + Child_elder + Pclass.Sex
+ Pclass.Child_elder, data=data_new, method="class")
printcp(fit5)
fit55 <- prune(fit5, cp= fit5$cptable[which.min(fit5$cptable[,"xerror"]),"CP"])
summary(fit55)
rpart.plot(fit55, branch=1, branch.type=2, type=2, extra=102,
shadow.col="gray", box.col="darkseagreen3",
border.col="darkseagreen3", split.col="chocolate4",
split.cex=1.2, main="Decision Tree of Titanic Survivals", uniform=T)