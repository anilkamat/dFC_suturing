SELECT 	*
FROM parks_and_recreation.employee_demographics;

SELECT 	first_name, last_name, birth_date
FROM parks_and_recreation.employee_demographics;

SELECT 	first_name, 
	last_name, 
	birth_date, 
    age, 
    (age+10)*10
FROM parks_and_recreation.employee_demographics;
# PEMDAS  # (Parentheses, explonents, mulitplication, division, addition and subtraction)

SELECT distinct first_name
FROM parks_and_recreation.employee_demographics;

select distinct first_name, gender 
from parks_and_recreation.employee_demographics







