SELECT min_salary, job_id
FROM jobs;

SELECT *
FROM employees
WHERE (salary, job_id)  in (SELECT min_salary, job_id
FROM jobs);