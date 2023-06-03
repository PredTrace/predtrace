import sys
sys.path.append("../../../")
from interface import *


"""
select
  c_custkey,
  c_name,
  sum(l_extendedprice * (1 - l_discount)) as revenue,
  c_acctbal,
  n_name,
  c_address,
  c_phone,
  c_comment
from
  customer,
  orders,
  lineitem,
  nation
where
  c_custkey = o_custkey k 
  and l_orderkey = o_orderkey k
  and o_orderdate >= '1993-10-01' (f) k
  and o_orderdate < '1994-01-01' (f) k
  and l_returnflag = 'R' (f) k
  and c_nationkey = n_nationkey (f)
group by
  c_custkey,
  c_name,
  c_acctbal,
  c_phone,
  n_name,
  c_address,
  c_comment
order by
  revenue desc,
  c_custkey
limit 20
;
"""



op1 = InitTable('data/orders.csv')

op4 = InitTable('data/lineitem.csv')

op6 = InitTable('data/customer.csv')
op7 = InitTable('data/nation.csv')
op2 = Filter(op1, BinOp(Field('o_orderdate'),'subset',Constant('1993-10-01')))
op3 = Filter(op2, BinOp(Field('o_orderdate'),'subset',Constant('1994-01-01')))
op5 = Filter(op4, BinOp(Field('l_returnflag'),'==',Constant('R')))

op8 = InnerJoin(op6, op3, ['c_custkey'],['o_custkey'])
op9 = InnerJoin(op5, op8, ['l_orderkey'],['o_orderkey']) 
op10 = InnerJoin(op9, op7, ['c_nationkey'],['n_nationkey'])


op11 = GroupBy(op5, ['c_custkey','c_name', 'c_acctbal', 'c_phone', 'n_name', 'c_address', 'c_comment'], \
    {
     'revenue':(Value(0), 'lambda row: row["l_extendedprice"]*(1-row["l_discount"])')}, 
	 {'revenue':'revenue'})

op12 = TopN(op11, 20, ['revenue','c_custkey'])
