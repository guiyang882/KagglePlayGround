options(stringsAsFactors=F,scipen=99)
rm(list=ls());gc()
require(data.table)

label_train <- fread("/Users/liuguiyang/Documents/CodeProj/PyProj/Kaggle/MobileUserDemographics/data/gender_age_train.csv",
                     colClasses=c("character","character",
                                  "integer","character"))
label_test <- fread("/Users/liuguiyang/Documents/CodeProj/PyProj/Kaggle/MobileUserDemographics/data/gender_age_test.csv",
                    colClasses=c("character"))
label_test$gender <- label_test$age <- label_test$group <- NA
label <- rbind(label_train,label_test)
setkey(label,device_id)
rm(label_test,label_train);gc()

brand <- fread("/Users/liuguiyang/Documents/CodeProj/PyProj/Kaggle/MobileUserDemographics/data/phone_brand_device_model.csv",
               colClasses=c("character","character","character"))
setkey(brand,device_id)
brand0 <- unique(brand,by=NULL)
brand0 <- brand0[sample(nrow(brand0)),]
brand2 <- brand0[-which(duplicated(brand0$device_id)),]
label1 <- merge(label,brand2,by="device_id",all.x=T)
rm(brand,brand0,brand2);gc()

# apps
events <- fread("/Users/liuguiyang/Documents/CodeProj/PyProj/Kaggle/MobileUserDemographics/data/events.csv",
                colClasses=c("character","character","character",
                             "numeric","numeric"))
setkeyv(events,c("device_id","event_id"))
event_app <- fread("/Users/liuguiyang/Documents/CodeProj/PyProj/Kaggle/MobileUserDemographics/data/app_events.csv",
                   colClasses=rep("character",4))
setkey(event_app,event_id)

events <- unique(events[,list(device_id,event_id)],by=NULL)
event_apps <- event_app[,list(apps=paste(unique(app_id),collapse=",")),by="event_id"]
device_event_apps <- merge(events,event_apps,by="event_id")
rm(events,event_app,event_apps);gc()

f_split_paste <- function(z){
  paste(unique(unlist(strsplit(z,","))),collapse=",")
}

device_apps <- device_event_apps[,list(apps=f_split_paste(apps)),by="device_id"]
rm(device_event_apps,f_split_paste);gc()

tmp <- strsplit(device_apps$apps,",")
device_apps <- data.table(device_id=rep(device_apps$device_id,
                                        times=sapply(tmp,length)),
                          app_id=unlist(tmp))
rm(tmp)

# dummy
d1 <- label1[,list(device_id,phone_brand)]
label1$phone_brand <- NULL
d2 <- label1[,list(device_id,device_model)]
label1$device_model <- NULL
d3 <- device_apps
rm(device_apps)

d1[,phone_brand:=paste0("phone_brand:",phone_brand)]
d2[,device_model:=paste0("device_model:",device_model)]
d3[,app_id:=paste0("app_id:",app_id)]

names(d1) <- names(d2) <- names(d3) <- c("device_id","feature_name")
dd <- rbind(d1,d2,d3)
rm(d1,d2,d3);gc()

require(Matrix)
ii <- unique(dd$device_id)
jj <- unique(dd$feature_name)

id_i <- match(dd$device_id,ii)
id_j <- match(dd$feature_name,jj)

id_ij <- cbind(id_i,id_j)
M <- Matrix(0,nrow=length(ii),ncol=length(jj),
            dimnames=list(ii,jj),sparse=T)
M[id_ij] <- 1
rm(ii,jj,id_i,id_j,id_ij,dd);gc()

x <- M[rownames(M) %in% label1$device_id,]
id <- label1$device_id[match(label1$device_id,rownames(x))]
y <- label1$group[match(label1$device_id,rownames(x))]
rm(M,label1)

# level reduction
x_train <- x[!is.na(y),]
tmp_cnt_train <- colSums(x_train)
x <- x[,tmp_cnt_train>0 & tmp_cnt_train<nrow(x_train)]
rm(x_train,tmp_cnt_train)


(group_name <- na.omit(unique(y)))
idx_train <- which(!is.na(y))
idx_test <- which(is.na(y))
train_data <- x[idx_train,]
test_data <- x[idx_test,]
train_label <- match(y[idx_train],group_name)-1
test_label <- match(y[idx_test],group_name)-1

require(xgboost)
dtrain <- xgb.DMatrix(train_data,label=train_label,missing=NA)
dtest <- xgb.DMatrix(test_data,label=test_label,missing=NA)

# param <- list(booster="gblinear",
#               num_class=length(group_name),
#               objective="multi:softprob",
#               eval_metric="mlogloss",
#               eta=0.01,
#               lambda=5,
#               lambda_bias=0,
#               alpha=2)

param <- list(booster="gbtree",
              num_class=length(group_name),
              objective="multi:softprob",
              eval_metric="mlogloss",
              eta=0.1,
              max_depth=8,
              subsample=0.75,
              silent=1)
watchlist <- list(train=dtrain)

ntree <- 350
set.seed(114)
fit_xgb <- xgb.train(params=param,
                     data=dtrain,
                     nrounds=ntree,
                     watchlist=watchlist,
                     verbose=1)
pred <- predict(fit_xgb,dtest)
pred_detail <- t(matrix(pred,nrow=length(group_name)))
res_submit <- cbind(id=id[idx_test],as.data.frame(pred_detail))
colnames(res_submit) <- c("device_id",group_name)
write.csv(res_submit,file="submit_v0_8_1.csv",row.names=F,quote=F)

