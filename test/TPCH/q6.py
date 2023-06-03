import sys
sys.path.append("../../../")
from interface import *

"""
select
  sum(l_extendedprice * l_discount) as revenue
from
  lineitem
where 
  l_shipdate >= '1994-01-01' (f) k 
  and l_shipdate < '1995-01-01' (f) k 
  and l_discount between 0.05 and 0.07 (f) k
  and l_quantity < 24 (f)
  ;
"""
op1 = InitTable('data/lineitem.csv')
op2 = Filter(op1, BinOp(Field('l_shipdate'),'>=',Constant('1994-01-01')))
op3 = Filter(op2, BinOp(Field('l_shipdate'),'<',Constant('1995-01-01')))
op4 = Filter(op3, BinOp(Field('l_discount'),'>=',Constant(0.05)))
op5 = Filter(op4, BinOp(Field('l_discount'),'<=',Constant(0.07)))
op6 = Filter(op5, BinOp(Field('l_quantity'),'<',Constant(24)))
op7 = AllAggregate(op6, Value(0), 'lambda v,row: v + (row["l_extendedprice"]*row["l_discount"])')
