-- SQLite for loading data and create database
--
-- Author: Wei-Ting Chen, You-Ren Fang, Guangfei Zhu
-- ------------------------------------------------------

-- Create table to store information of items
drop table if exists Items;
create table Items(ItemID INTEGER PRIMARY KEY, Name VARCHAR(200), Category VARCHAR(100), CategoryLength INTEGER, Currently INTEGER, BuyPrice INTEGER, FirstBid INTEGER, NumberBid INTEGER, Started VARCHAR(30), Ends VARCHAR(30), SellerID VARCHAR(30), Description VARCHAR(1000));

-- Create table to store information of sellers
drop table if exists Sellers;
create table Sellers(UserID VARCHAR(30) PRIMARY KEY, Rating INTEGER, Location VARCHAR(30), Country VARCHAR(30));

-- Create table to store information of bidders
drop table if exists Bidders;
create table Bidders(UserID VARCHAR(30) PRIMARY KEY, Rating INTEGER, Location VARCHAR(30), Country VARCHAR(30));

-- Create table to establish relationship between bid, items, bidder and seller
drop table if exists Bid;
create table Bid(ItemID INTEGER, BidderID VARCHAR(30), SellerID VARCHAR(30), Amount FLOAT, Time VARCHAR(30),
		 PRIMARY KEY(ItemID, BidderID, SellerID)
		 FOREIGN KEY(ItemID)
		     REFERENCES Items(ItemID),
		 FOREIGN KEY(BidderID)
		     REFERENCES Bidders(UserID),
		 FOREIGN KEY(SellerID)
		     REFERENCES Sellers(UserID));

drop table if exists CategoryUserID;
create table CategoryUserID(Category VARCHAR(40), ItemID INTEGER);

