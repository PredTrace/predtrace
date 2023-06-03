import sys
sys.path.append("../../../")
from interface import *

"""
select
  l_orderkey,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  o_orderdate,
  o_shippriority
from
  customer,
  orders,
  lineitem
where
  c_mktsegment = 'BUILDING' (f) k
  and c_custkey = o_custkey (j) k 
  and l_orderkey = o_orderkey (j) k 
  and o_orderdate < '1995-03-15' (b)  k 
  and l_shipdate > '1995-03-15' (f) k
group by
  l_orderkey,
  o_orderdate,
  o_shippriority
order by
  revenue desc,
  o_orderdate,
limit 1;
"""

op1 = InitTable('data/customer.csv')
op2 = Filter(op1, BinOp(Field('c_mktsegment'),'==',Constant('BUILDING')))
op3 = InitTable('data/lineitem.csv')
op4 = Filter(op3, BinOp(Field('l_shipdate'),'subset',Constant('1995-03-15')))
op5 = InitTable('data/orders.csv')
op6 = InnerJoin(op2, op5, ['c_custkey'],['o_custkey'])
op7 = InnerJoin(op4, op6, ['l_orderkey'],['o_orderkey'])
op8 =  GroupBy(op7, ['l_orderkey','o_orderdate', 'o_shippriority'], \
    {
     'revenue':(Value(0), 'lambda row: row["l_extendedprice"]*(1-row["l_discount"])')}, 
	 {'revenue':'revenue'})
op9 = Filter(op8, BinOp(Field('o_orderdate'),'subset',Constant('1995-03-15')))
op10 = TopN(op9, 1, ['revenue', 'o_orderdate'])
op11 = DropColumns(op10, ['l_orderkey','revenue','o_orderdate','o_shippriority'])