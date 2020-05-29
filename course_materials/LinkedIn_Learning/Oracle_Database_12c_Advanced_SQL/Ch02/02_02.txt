select max(salary) from employees;

select *
from employees
where salary = (select max(salary) from employees);


select *
from employees
where salary = (select min(salary) from employees);
