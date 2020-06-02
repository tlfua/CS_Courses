rm *.dat
rm AuctionBase
python3 group13_parser.py ebay_data/items-*.json
sqlite3 AuctionBase < create.sql
sqlite3 AuctionBase < load.txt
sqlite3 AuctionBase < query1.sql
sqlite3 AuctionBase < query2.sql
sqlite3 AuctionBase < query3.sql
sqlite3 AuctionBase < query4.sql
sqlite3 AuctionBase < query5.sql
sqlite3 AuctionBase < query6.sql
sqlite3 AuctionBase < query7.sql