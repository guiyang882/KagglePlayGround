SELECT * FROM db_mobileuser.t_prepare_train;

select 
device_id, age, count(*) 
from t_prepare_train 
where age >= 0 and age < 20
group by device_id, age;
