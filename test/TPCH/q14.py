import sys
sys.path.append("../../../")
from interface import *

"""
select
  100.00 * sum(case
    when p_type like 'PROMO%'
    then l_extendedprice * (1 - l_discount)
    else 0.0
    end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
  lineitem,
  part
where
  l_partkey = p_partkey k
  and l_shipdate >= '1995-09-01' k
  and l_shipdate < '1995-10-01' k
"""

op1 = InitTable('data/part.csv')
op2 = InitTable('data/lineitem.csv')
op3 = Filter(op2, BinOp(Field('l_shipdate'),'subset',Constant('1995-09-01')))
op4 = Filter(op3, BinOp(Field('l_shipdate'),'subset',Constant('1995-10-01')))
op5 = InnerJoin(op1,op4, ['p_partkey'],['l_partkey'])

op6 =  SetItem(op5, 'promo_revenue', 'lambda row: 100.00 * sum(row["l_extendedprice"] * (1 - row["l_discount"])) if "PROMO" in row["p_type"] else 0')
op7 = SetItem(op6, 'promo_revenue', 'lambda row: row["promo_revenue"]/ sum(row["l_extendedprice"] * (1 - row["l_discount"]))')