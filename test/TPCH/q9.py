import sys
sys.path.append("../../../")
from interface import *

"""
select
  nation,
  o_year,
  sum(amount) as sum_profit
from(
  select
    n_name as nation,
    year(o_orderdate) as o_year,
    l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
  from
    part,
    supplier,
    lineitem,
    partsupp,
    orders,
    nation
  where
    s_suppkey = l_suppkey k
    and ps_suppkey = l_suppkey k
    and ps_partkey = l_partkey k 
    and p_partkey = l_partkey k
    and o_orderkey = l_orderkey k 
    and s_nationkey = n_nationkey k
    and p_name like '%green%' (f) k
  ) as profit
group by
  nation,
  o_year
order by
  nation,
  o_year desc
;
"""

op1 = InitTable('data/part.csv')
op3 = InitTable('data/supplier.csv')
op4 = InitTable('data/lineitem.csv')
op5 = InitTable('data/partsupp.csv')
op6 = InitTable('data/orders.csv')
op7 = InitTable('data/nation.csv')
op2 = Filter(op1, BinOp(Field('p_name'),'==',Constant('green')))
op8 = InnerJoin(op3, op4, ['s_suppkey'],['l_suppkey'])
op9 = InnerJoin(op5, op8, ['ps_suppkey', 'ps_partkey'],['l_suppkey', 'l_partkey'])
op10 = InnerJoin(op2, op9, ['p_partkey'],['l_partkey'])
op11 = InnerJoin(op6, op10, ['o_orderkey'],['l_orderkey'])
op12 = InnerJoin(op11, op7, ['s_nationkey'],['n_nationkey'])

op13 = SetItem(op12, 'amount', 'lambda row: row["l_extendedprice"] * (1 - row["l_discount"]) - row["ps_supplycost"] * row["l_quantity"]')
op14 = SetItem(op13, 'nation', 'lambda row: row["n_name"]')
op15 = SetItem(op14, 'o_year', 'lambda row: year(row["o_orderdate"])')
op16 = GroupBy(op15, ['nation', 'o_year'], \
    {
     'amount':(Value(0), 'sum')}, 
	 {'amount':'sum_profit'}
)
op17 = SortValues(op16, ['nation', 'o_year'])
