"""
Author: Varun Shah

AdWords Placement Problem via Online Bipartite Graph Matching

Description:
Each advertiser has a budget 𝐵𝑖. The bid of advertiser 𝑖 for a query 𝑞 is denoted as 𝑏𝑖𝑞.

Goal: maximize the amount of money received from the advertisers.

advertisers === bidders

Assumptions:
  The bids are small with respect to the daily budgets (𝑏𝑖𝑞 ≪ 𝐵𝑖). 
  Each advertisement slot can be allocated to at most one advertiser.
  Each ad request has one advertisement slot.

Datasets: 
1. bidder_dataset.csv
    advertiser_id, keyword (query), bid value, total budget
2. queries.txt
    query strings
"""
import pandas as pd
import math
from operator import itemgetter
import random
import sys

def read_filter_data():
    """
    Returns: bidder_df, bidder_budget_df, queries

    bidder_df :  index = 'Keyword', columns = [ 'Advertiser','Bid Value'  ]
    bidder_budget_df: index = 'Advertiser', columns = ['Budget']

    Read in bidder dataset and set index as keyword as primary searching will be on the keywords.
    Extract bidder budgets to another df.
    Remove budgets form bidder_df to avoid any potential silly mistakes.

    advertiser_id, keyword (query), bid value, total budget
    """
    bidder_df = pd.read_csv("./bidder_dataset.csv")
    bidder_budget_df = bidder_df.dropna()[['Advertiser','Budget']].set_index('Advertiser')
    bidder_df = bidder_df.set_index('Keyword').drop(columns = "Budget")
    # print(bidder_df.head(5)) 
    # print(bidder_budget_df.head(5))

    with open("queries.txt") as f:
        # read().splitlines() removes trailing \n while readlines does not.
        queries = f.read().splitlines()
    #print(queries)

    return bidder_df, bidder_budget_df, queries

def find_interested_bidder_budget_df(interested_bidders_df,bidder_budget_df):
    """
    keeps only those bidders in bidder_budget_df which are also present in interested_bidders_df
    reset_index moves index to column
    """
    interested_bidder_list = [ row["Advertiser"] for index, row in interested_bidders_df.iterrows() ]
    bidder_budget_df =  bidder_budget_df.reset_index()
    interested_bidder_budget_df = bidder_budget_df[bidder_budget_df["Advertiser"].isin(interested_bidder_list)]
    return interested_bidder_budget_df

def balance(interested_bidders,bidder_budget_df):
    """
    Returns  designated_bidder, revenue
    Performs balance matching of query with interested bidder --
        match query to the interested bidder with the highest unspent budget.

    Keep only budgets of interested bidders.
    Sort interested bidders acc to their unspent budget in descending order. 
        In case of a tie, sort ascending by advertiser_id

    reset_index adds index to column.
    
    Return bidder with highest unspent budget and revenue gained.
    """
    
    interested_bidder_budget_df = find_interested_bidder_budget_df(interested_bidders,bidder_budget_df) 
    sorted_interested_bidder_budget_df = interested_bidder_budget_df.sort_values(by=["Budget","Advertiser"],ascending=[False,True])

    for index, row in  sorted_interested_bidder_budget_df.iterrows():
        bidder = interested_bidders.loc[ interested_bidders["Advertiser"] == row["Advertiser"] ] 
        bid = bidder.iloc[0,1]
        if( row["Budget"] >= bid ):
            return row["Advertiser"],bid

    return None,0

def greedy(interested_bidders,bidder_budget_df):
    """
    Returns  designated_bidder, revenue
    Performs greedy matching of query with interested bidder -- match query to the interested bidder with the highest bid.

    Sort interested bidders acc to their bid values in descending order.
    In case of a tie, sort ascending by advertiser_id
    Return highest bidder with enough budget.
    """
    sorted_interested_bidders = interested_bidders.sort_values(by=["Bid Value","Advertiser"],ascending=[False,True])
    for index, row in sorted_interested_bidders.iterrows():
        assigned_bidder = bidder_budget_df.loc[row["Advertiser"],:]
        if(assigned_bidder.iloc[0] >= row["Bid Value"]):
            return row["Advertiser"],row["Bid Value"]

    return None,0

def msvv(interested_bidders,bidder_budget_df,orig_bidder_budget_df):
    """
    Returns  designated_bidder, revenue
    Performs msvv matching of query with interested bidder -- 
        match query to the interested bidder with the highest bid*psi value
            where psi = (1 - e^(fraction_budget_spent -1))

    Sort interested bidders acc to their bid values in descending order.
    In case of a tie, sort ascending by advertiser_id
    Return highest bidder with enough budget.

    reset_index adds index to column.

    bidder_df :  index = 'Keyword', columns = [ 'Advertiser','Bid Value'  ]
    bidder_budget_df: index = xxx, columns = ['Budget','Advertiser']
    """

    interested_bidder_budget_df = find_interested_bidder_budget_df(interested_bidders,bidder_budget_df) 
    bidder_scale_list = []
    interested_bidders = interested_bidders.reset_index().set_index("Advertiser")
    for index, row in interested_bidder_budget_df.iterrows():
        original_budget = orig_bidder_budget_df.loc[row["Advertiser"]].iloc[0]
        fraction_of_budget_spent =  (original_budget - row["Budget"]) / original_budget
        psi = 1 - math.exp( (fraction_of_budget_spent-1) )
        original_bid = interested_bidders.loc[ row["Advertiser"] ].iloc[1]
        scaled_bid = original_bid * psi
        bidder_scale_list.append([row["Advertiser"], scaled_bid])
    
    sorted_bidder_scale_list = sorted(bidder_scale_list,key=itemgetter(1),reverse = True)
    
    for l in sorted_bidder_scale_list:
        bidder = l[0]
        bid = interested_bidders.loc[bidder].iloc[1]
        budget = interested_bidder_budget_df.loc[bidder].iloc[1]
        if(budget >= bid):
            return bidder,bid

    return None,0

def adwords(bidder_df, bidder_budget_df, queries, algo):
    """
    Returns the adwords revenue
    For each query 𝑞
        If all neighbors (bidding advertisers for 𝑞) have spent their full budgets
            continue
        Else
            perform appropriate operation
    """
    total_revenue = 0
    orig_bidder_budget_df = bidder_budget_df.copy()
    for query in queries:
        interested_bidders = bidder_df.loc[[query]]
        if(algo == "greedy"):
            designated_bidder,revenue = greedy(interested_bidders,bidder_budget_df)
        if(algo == "balance"):
            designated_bidder,revenue = balance(interested_bidders,bidder_budget_df)
        if(algo == "msvv"):
            designated_bidder,revenue = msvv(interested_bidders,bidder_budget_df,orig_bidder_budget_df)
        
        # reduce designated bidder's budget
        if(designated_bidder != None):
            budget = bidder_budget_df.loc[designated_bidder,"Budget"]
            bidder_budget_df.loc[designated_bidder,"Budget"] = (budget - revenue)
        total_revenue += revenue

    return total_revenue

def competitive_ratio(bidder_df, bidder_budget_df, queries, algo, optimal_revenue):
    """
    returns competitive ratio.
    Competitive ratio is defined as mean revenue over 100 input permutations divided by optimal_revenue
    """
    random.seed(0)
    total_revenue = 0.0
    permutations = 100
    for i in range(permutations):
        random.shuffle(queries)
        total_revenue+=adwords(bidder_df, bidder_budget_df, queries, algo=algo)
    
    avg_revenue = total_revenue / permutations
    ratio = avg_revenue/optimal_revenue
    return ratio

def main():
    # Read data
    bidder_df, bidder_budget_df, queries = read_filter_data()
    #Optimal revenue is assumed to be sum of all budgets.
    optimal_revenue = bidder_budget_df.sum(axis=0).iloc[0]

    algo=''
    if sys.argv[1] == 'greedy':
        algo='greedy'
    elif sys.argv[1] == 'msvv':
        algo='msvv'
    elif sys.argv[1] == 'balance':
        algo='balance'
    else:
        print('invalid input')
        sys.exit(0)

    total_revenue = adwords(bidder_df, bidder_budget_df, queries, algo=algo)
    print(total_revenue)
    cr = competitive_ratio(bidder_df, bidder_budget_df, queries, algo, optimal_revenue)
    print(cr)


main()
