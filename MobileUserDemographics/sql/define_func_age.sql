DELIMITER //

DROP PROCEDURE IF EXISTS splitage //

CREATE PROCEDURE 
  splitage( low_age int, high_age int , sex varchar(3) )
BEGIN  
	select *, count(*) as cnt
	from t_prepare_train 
	where age >= low_age and age < high_age and gender = sex
	group by device_id, age; 
END 
//

DELIMITER ;

call splitage(00,20, 'M');
call splitage(20,30, 'M');
call splitage(30,40, 'M');
call splitage(40,50, 'M');
call splitage(50,70, 'M');
call splitage(70,120, 'M');
