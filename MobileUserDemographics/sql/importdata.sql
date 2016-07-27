SELECT * FROM db_mobileuser.t_prepare_app_events;
select * from db_mobileuser.t_prepare_train;

LOAD DATA INFILE  '/home/fighter/Kaggle/MobileUserDemographics/data/prepare_train.csv' 
INTO TABLE db_mobileuser.t_prepare_train 
COLUMNS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' ESCAPED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA INFILE  '/home/fighter/Kaggle/MobileUserDemographics/data/prepare_app_events.csv' 
INTO TABLE db_mobileuser.t_prepare_app_events 
COLUMNS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' ESCAPED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA INFILE  '/home/fighter/Kaggle/MobileUserDemographics/data/prepare_app.csv' 
INTO TABLE db_mobileuser.t_prepare_app
COLUMNS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' ESCAPED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;