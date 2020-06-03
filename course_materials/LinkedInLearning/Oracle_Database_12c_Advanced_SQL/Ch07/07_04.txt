CREATE TABLE sales_list
(
salesrep_id  NUMBER(5), 
	salesrep_name VARCHAR2(40),
	sales_state   VARCHAR2(30),
	sales_value  NUMBER(10), 
	sales_date    DATE)
PARTITION BY LIST(sales_state)
(
PARTITION sales_CA VALUES('CA'),
PARTITION sales_NY VALUES ('NY'),
PARTITION sales_central VALUES('TX', 'IL'),
PARTITION sales_other VALUES(DEFAULT)
);

INSERT INTO sales_list VALUES  (100, 'Picard', 'CA', 100, '01-JAN-2017');

INSERT INTO sales_list VALUES  (200, 'Janeway', 'NY', 500, '02-JAN-2017');

INSERT INTO sales_list VALUES  (300, 'Kirk', 'TX', 1000, '03-JAN-2017');
INSERT INTO sales_list VALUES  (100, 'Picard', 'IL', 500, '04-JAN-2017');

select * from sales_list;


SELECT COUNT(*) FROM sales_list partition (sales_ca);
SELECT COUNT(*) FROM sales_list partition (sales_ny);
SELECT COUNT(*) FROM sales_list partition (sales_central);
SELECT COUNT(*) FROM sales_list partition (sales_other);


INSERT INTO sales_list VALUES  (200, 'Janeway', 'FL', 999, '05-JAN-2017');


SELECT COUNT(*) FROM sales_list partition (sales_other);