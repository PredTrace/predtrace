import sys
sys.path.append("../../../")
from interface import *

"""
select
  n_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue
from
  customer,
  orders,
  lineitem,
  supplier,
  nation,
  region
where
  c_custkey = o_custkey k
  and l_orderkey = o_orderkey k
  and l_suppkey = s_suppkey k
  and c_nationkey = s_nationkey k
  and s_nationkey = n_nationkey k
  and n_regionkey = r_regionkey k
  and r_name = 'ASIA' (f) k
  and o_orderdate >= '1994-01-01' (f) k
  and o_orderdate < '1995-01-01' (f) k 
group by
  n_name
order by
  revenue desc
;
"""
op1 = InitTable('data/region.csv')
op3 = InitTable('data/orders.csv')
op6 = InitTable('data/customer.csv')
op7 = InitTable('data/lineitem.csv')
op8 = InitTable('data/supplier.csv')
op9 = InitTable('data/nation.csv')
op10 = Filter(op3, BinOp(Field('o_orderdate'),'subset',Constant('1994-01-01')))
op11 = Filter(op10, BinOp(Field('o_orderdate'),'subset',Constant('1995-01-01')))
op12 = Filter(op1, BinOp(Field('r_name'),'==',Constant('ASIA')))

op13 = InnerJoin(op9, op12, ['n_regionkey'],['r_regionkey'])
op14 = InnerJoin(op6, op13, ['c_nationkey'],['n_nationkey'])
op15 = InnerJoin(op11, op14, ['o_custkey'],['c_custkey'])
op16 = InnerJoin(op8, op15, ['s_nationkey'],['c_nationkey'])
op17 = InnerJoin(op7, op16, ['l_suppkey'],['s_suppkey'])
op18 = Filter(op17, BinOp(Field('c_nationkey'), '==', Field('s_nationkey')))


op19 = GroupBy(op18, ['n_name'], \
    {
     'revenue':(Value(0), 'lambda row: row["l_extendedprice"]*(1-row["l_discount"])')}, 
	 {'revenue':'revenue'})
op20 = SortValues(op19, ['revenue'])