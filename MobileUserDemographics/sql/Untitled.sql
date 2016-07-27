select DAYOFWEEK(timestamp),hour(timestamp) from t_prepare_train;

select count(*) from t_prepare_train;

select 
device_id, gender, age, event_id, 
DAYOFWEEK(timestamp) as do_weekday ,hour(timestamp) as do_hour, 
longtitude, latitude, phone_brand, group_id 
from t_prepare_train;

select 
device_id, gender, age, 
DAYOFWEEK(timestamp) as do_weekday ,hour(timestamp) as do_hour, 
phone_brand, event_id, group_id 
from t_prepare_train;

select 
device_id, gender, age, 
DAYOFWEEK(timestamp) as do_weekday ,hour(timestamp) as do_hour, 
phone_brand, event_id, group_id 
from t_prepare_train;

SELECT 
event_id, general_group, count(*) as cnt
FROM db_mobileuser.t_prepare_app_events
where is_installed = 1 and is_active = 1
group by event_id, general_group;

select * from t_prepare_app limit 10;
select distinct(general_group) from t_prepare_app;

select distinct(event_id) from t_prepare_app;

select general_group, cnt from t_prepare_app where event_id = '1000000';

select count(*) from db_mobileuser.t_prepare_train;

select device_id, DAYOFWEEK(timestamp) as do_weekday, hour(timestamp) as do_hour, longtitude, latitude, event_id from t_prepare_train;

select longtitude, latitude from t_prepare_train;
