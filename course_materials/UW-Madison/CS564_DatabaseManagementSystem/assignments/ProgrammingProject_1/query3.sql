WITH Temporary(count) AS (SELECT COUNT(Category) FROM CategoryUserID GROUP BY ItemID) SELECT COUNT(count) FROM Temporary WHERE count= 4;
