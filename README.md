# AdWords Placement Problem via Online Bipartite Graph Matching ( using Greedy, balanced and msvv algos )

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
    
1. Greedy:
  Tries to match query to the highest bid.
  Works well when bids are highly skewed.
  
2. Balance:
  Tries to match query to neighbor with highest unspent budget.
  Works when bids are uniform.
  
3. MSVV mixes greedy and balance.
  Gives best of both worlds.
