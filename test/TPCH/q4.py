import sys
sys.path.append("../../../")
from interface import *


"""
select
  o_orderpriority,
  count(*) as order_count
from
  orders
where
  o_orderdate >= '1993-07-01'
  and o_orderdate < '1993-10-01'
  and exists (
    select
      *
    from
      lineitem
    where
      l_orderkey = o_orderkey
      and l_commitdate < l_receiptdate
    )
group by
  o_orderpriority
order by
  o_orderpriority
;
"""
op1 = InitTable('data/lineitem.csv')
#op2 = Filter(op1, BinOp(Field('l_commitdate'), '<', Field('l_receiptdate')))
op2 = InitTable('data/orders.csv')

sub_op_row = SubpipeInput(op2, 'row')
sub_op1 = SubpipeInput(op1, 'table')
temp = ScalarComputation({'row':sub_op_row}, 'lambda row: row["o_orderkey"]')
sub_op2 = Filter(sub_op1, BinOp(Field('l_commitdate'), '<', Field('l_receiptdate')))
sub_op3 = Filter(sub_op2, BinOp(Field('l_orderkey'), '==', temp))
count = AllAggregate(sub_op3, Value(0), 'lambda v,row: 1')

op3 = CrosstableUDF(op2, 'exists_count', SubPipeline(PipelinePath([sub_op_row, sub_op1, temp, sub_op2, sub_op3, count])))
op4 = GroupBy(op3, ['o_orderpriority'], \
    {
      'order_count':(Value(0),'count')}, 
	 {'order_count':'order_count'})
op5 = SortValues(op4, ["o_orderpriority"])

# exist_cond = InnerJoin(op2, op3, ["l_orderkey"], ["o_orderkey"])
# cond = ScalarComputation({'count':exist_cond}, 'lambda count: count>0')
# op4 = Filter(op3, BinOp(Field('o_orderdate'), '>=', Constant('1993-07-01')))
# op5 = Filter(op4, BinOp(Field('o_orderdate'), '<', Constant('1993-10-01')))
# # Yin: take into account the condition
# op6 = Filter(op5, BinOp(cond, '>',Constant(0)))
# op7 = GroupBy(op6, ['o_orderpriority'], \
#     {
#       'count_order':(Value(0),'count')}, 
# 	 {'count_order':'count_order'})
# op8 = SortValues(op6, ["o_orderpriority"])