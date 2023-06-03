import sys
sys.path.append("../../../")
from interface import *

"""
select
  o_year,
  sum(case
    when nation = 'BRAZIL'
    then volume
    else 0
  end) / sum(volume) as mkt_share
from (
  select
    year(o_orderdate) as o_year,
    l_extendedprice * (1 - l_discount) as volume,
    n2.n_name as nation
  from
    part,
    supplier,
    lineitem,
    orders,
    customer,
    nation n1,
    nation n2,
    region
  where
    p_partkey = l_partkey k
    and s_suppkey = l_suppkey k 
    and l_orderkey = o_orderkey k 
    and o_custkey = c_custkey k
    and c_nationkey = n1.n_nationkey k 
    and n1.n_regionkey = r_regionkey k 
    and r_name = 'AMERICA' (f)  k
    and s_nationkey = n2.n_nationkey k
    and o_orderdate between '1995-01-01' and '1996-12-31' (f) k 
    and p_type = 'ECONOMY ANODIZED STEEL'  (f) k
  ) as all_nations
group by
  o_year
order by
  o_year
"""

region = InitTable('data/region.csv')

orders = InitTable('data/orders.csv')

part = InitTable('data/part.csv')

lineitem = InitTable('data/lineitem.csv')

op2 = Filter(region, BinOp(Field('r_name'),'==',Constant('AMERICA')))
op4 = Filter(orders, BinOp(Field('o_orderdate'),'subset',Constant('1995-01-01')))
op5 = Filter(op4, BinOp(Field('o_orderdate'),'subset',Constant('1996-12-31')))
op7 = Filter(part, BinOp(Field('p_type'),'==',Constant('ECONOMY ANODIZED STEEL')))

op9 = InnerJoin(op7, lineitem, ["p_partkey"], ["l_partkey"])
supplier = InitTable('data/supplier.csv')
op11 = InnerJoin(supplier, op9, ["s_suppkey"],["l_suppkey"])
op12 = InnerJoin(op11, op5, ["l_orderkey"],["o_orderkey"])
customer = InitTable('data/customer.csv')
op14 = InnerJoin(op12, customer,["o_custkey"],["c_custkey"])
nation1 = InitTable('data/nation.csv')
nation2 = InitTable('data/nation.csv')
op16 = InnerJoin(op14, nation1,["c_nationkey"],["n_nationkey"])
op18 = InnerJoin(op16, op2,["n_regionkey"],["r_regionkey"])
op19 = InnerJoin(nation2, op18,["n_nationkey"],["s_nationkey"])



op20 = SetItem(op19, 'volume', 'lambda row: row["l_extendedprice"]*(1-row["l_discount"])')
op21 = SetItem(op20, 'nation', 'lambda row: row["n_name_x"]')
op22 = SetItem(op21, 'o_year', 'lambda row: year(row["o_orderdate"])')
op23 = GroupBy(op22, ['o_year'], \
    {
     'mkt_share':(Value(0), 'lambda row: row["volume"] if row["nation"] = "BRAZIL" else 0')}, 
	 {'mkt_share':'mkt_share'})
op24 = SetItem(op23, 'mkt_share', 'lambda row: row["mkt_share"]/row["volume"]')
op25 = SortValues(op24, ["o_year"])