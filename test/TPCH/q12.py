import sys
sys.path.append("../../../")
from interface import *


"""
select
  l_shipmode,
  sum(case
    when o_orderpriority = '1-URGENT'
      or o_orderpriority = '2-HIGH'
    then 1
    else 0
  end) as high_line_count,
  sum(case
    when o_orderpriority <> '1-URGENT'
      and o_orderpriority <> '2-HIGH'
    then 1
    else 0
  end) as low_line_count
from
  orders,
  lineitem
where
  o_orderkey = l_orderkey
  and l_shipmode in ('MAIL', 'SHIP') (b)
  and l_commitdate < l_receiptdate (f) k 
  and l_shipdate < l_commitdate (f) k 
  and l_receiptdate >= '1994-01-01' (f) k
  and l_receiptdate < '1995-01-01' (f) k
group by
  l_shipmode
order by
  l_shipmode
"""

op1 = InitTable('data/lineitem.csv')
op6 = InitTable('data/orders.csv')
op2 = Filter(op1, BinOp(Field('l_commitdate'), 'subset', Field('l_receiptdate')))
op3 = Filter(op2, BinOp(Field('l_shipdate'), 'subset', Field('l_commitdate')))
op4 = Filter(op3, BinOp(Field('l_receiptdate'), 'subset', Constant('1994-01-01')))
op5 = Filter(op4, BinOp(Field('l_receiptdate'), 'subset', Constant('1995-01-01')))
op10 = Filter(op5, BinOp(Field('l_shipmode'), 'in', Constant(['MAIL', "SHIP"])))
op7 = InnerJoin(op6, op10, ["o_orderkey"],["l_orderkey"])
op8 = GroupBy(op7, ['l_shipmode'], \
    {
      'high_line_count':(Value(0),'lambda row: 1 if row["o_orderpriority"] = "1-URGENT" or row["o_orderpriority"] = "2-HIGH" else 0'), \
      'low_line_count':(Value(0),'lambda row: 1 if row["o_orderpriority"] <> "1-URGENT" or row["o_orderpriority"] <> "2-HIGH" else 0'),}, 
	 {'high_line_count':'high_line_count', 'low_line_count':'low_line_count'})
op9 = SortValues(op8, ["l_shipmode"])
