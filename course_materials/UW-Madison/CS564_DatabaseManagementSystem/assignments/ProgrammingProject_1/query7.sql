SELECT COUNT(DISTINCT C.Category) FROM CategoryUserID AS C, Bid WHERE Bid.Amount > 100 AND Bid.ItemID = C.ItemID;