
import sys
from json import loads
from re import sub
import pandas as pd
import os

columnSeparator = "|"

# Dictionary of months used for date transformation
MONTHS = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',\
        'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}

"""
Returns true if a file ends in .json
"""
def isJson(f):
    return len(f) > 5 and f[-5:] == '.json'

"""
Converts month to a number, e.g. 'Dec' to '12'
"""
def transformMonth(mon):
    if mon in MONTHS:
        return MONTHS[mon]
    else:
        return mon

"""
Transforms a timestamp from Mon-DD-YY HH:MM:SS to YYYY-MM-DD HH:MM:SS
"""
def transformDttm(dttm):
    dttm = dttm.strip().split(' ')
    dt = dttm[0].split('-')
    date = '20' + dt[2] + '-'
    date += transformMonth(dt[0]) + '-' + dt[1]
    return date + ' ' + dttm[1]

"""
Transform a dollar value amount from a string like $3,453.23 to XXXXX.xx
"""
def transformDollar(money):
    if money == None or len(money) == 0:
        return money
    return sub(r'[^\d.]', '', money)

"""
Created by Guangfei Zhu:
    This function is used to remove duplicates in the list
"""
def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

"""
Parses a single json file. Currently, there's a loop that iterates over each
item in the data set. Your job is to extend this functionality to create all
of the necessary SQL tables for your database.
"""
def parseJson(json_file):
    with open(json_file, 'r') as f:
        items = loads(f.read())['Items'] # creates a Python dictionary of Items for the supplied json file
        itemAttribute = []
        bidAttribute = []
        sellerAttribute = []
        bidderAttribute = []
        categoryUserID = []

        for item in items:
            itemID = int(item['ItemID'])
            name = item['Name']
            category = item['Category']
            for catei in category:
                category_tup = (catei, itemID)
                categoryUserID.append(category_tup)
            currently = transformDollar(item['Currently'])
            firstBid = transformDollar(item['First_Bid'])
            numBid = item['Number_of_Bids']
            started = transformDttm(item['Started'])
            ends = transformDttm(item['Ends'])
            sellerID = item['Seller']['UserID']
            if 'Buy_Price' in item.keys():
                buy_price = transformDollar(item['Buy_Price'])
            else:
                buy_price = 'None'
            description = item['Description']

            item_tup = (itemID, name, category, len(category), currently,buy_price, firstBid, numBid, started,ends,sellerID, description)
            itemAttribute.append(item_tup)

            seller_userid = item['Seller']['UserID']
            seller_rating = item['Seller']['Rating']
            seller_location  = item['Location']
            seller_country = item['Country']

            seller_tup = (seller_userid, seller_rating, seller_location, seller_country)
            sellerAttribute.append(seller_tup)

            if item['Bids'] != None:
                for bid in item['Bids']:
                    bidder_userid = bid['Bid']['Bidder']['UserID']
                    if 'Location' in bid['Bid']['Bidder'].keys():
                        bidder_location = bid['Bid']['Bidder']['Location']
                    else:
                        bidder_location = 'None'
                    if 'Country' in bid['Bid']['Bidder'].keys():
                        bidder_country = bid['Bid']['Bidder']['Country']
                    else:
                        bidder_country = 'None'
                    bidder_rating = bid['Bid']['Bidder']['Rating']
                    bidder_tup = (bidder_userid, bidder_rating, bidder_location, bidder_country)
                    bidderAttribute.append(bidder_tup)

                    bid_amount = transformDollar(bid['Bid']['Amount'])
                    bid_time = transformDttm(bid['Bid']['Time'])
                    bid_tup = (itemID, bidder_userid, seller_userid, bid_amount, bid_time)
                    bidAttribute.append(bid_tup)

        removed_list1 = list(uniq(sorted(itemAttribute)))
        removed_list2 = list(uniq(sorted(sellerAttribute)))
        removed_list3 = list(uniq(sorted(bidderAttribute)))
        removed_list4 = list(uniq(sorted(bidAttribute)))
        removed_list5 = list(uniq(sorted(categoryUserID)))

        df1 = pd.DataFrame(removed_list1)
        df2 = pd.DataFrame(removed_list2)
        df3 = pd.DataFrame(removed_list3)
        df4 = pd.DataFrame(removed_list4)
        df5 = pd.DataFrame(removed_list5)

        df1.to_csv(json_file[10:-5] + '_item.dat', index=False, header=False)
        df2.to_csv(json_file[10:-5] + '_seller.dat', index=False, header=False)
        df3.to_csv(json_file[10:-5] + '_bidder.dat', index=False, header=False)
        df4.to_csv(json_file[10:-5] + '_bid.dat', index=False, header=False)
        df5.to_csv(json_file[10:-5] + '_categoryUserID.dat', index=False, header=False)

"""
Loops through each json files provided on the command line and passes each file
to the parser
"""
def main(argv):
    if len(argv) < 2:
        print >> sys.stderr, 'Usage: python skeleton_json_parser.py <path to json files>'
        sys.exit(1)
    # loops over all .json files in the argument
    for f in argv[1:]:
        if isJson(f):
            parseJson(f)
            print (("Success parsing " + f))

if __name__ == '__main__':
    main(sys.argv)

