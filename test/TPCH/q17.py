import sys
sys.path.append("../../../")
from interface import *

"""
select
  sum(l_extendedprice) / 7.0 as avg_yearly
from
  lineitem,
  part
where
  p_partkey = l_partkey
  and p_brand = 'Brand#23' (f) k 
  and p_container = 'MED BOX' (f) k 
  and l_quantity < (
    select
      0.2 * avg(l_quantity)
    from
      lineitem
    where
      l_partkey = p_partkey
  )
"""
part = InitTable('data/part.csv')
lineitem = InitTable('data/lineitem.csv')
op1 = InnerJoin(part, lineitem, ['p_partkey'],['l_partkey'])
op2 = Filter(op1, BinOp(Field('p_brand'),'==',Constant('Brand#23')))
op3 = Filter(op2, BinOp(Field('p_container'),'==',Constant('MED BOX')))

sub_op_row = SubpipeInput(op3, 'row')
sub_op_table = SubpipeInput(lineitem, 'table')

temp = ScalarComputation({'x':sub_op_row},'lambda x: x["p_partkey"]')
sub_op1 = Filter(sub_op_table, BinOp(Field('l_partkey'), '==', temp))
sub_op_sum = AllAggregate(sub_op1, Value(0), 'lambda v,row: v + row["l_quantity"]')
sub_op_cnt = AllAggregate(sub_op1, Value(0), 'lambda v,row: v + 1')
sub_op_avg = ScalarComputation({'s':sub_op_sum, 'c':sub_op_cnt},'lambda s, c: 0.2 * (s/c)')
op4 = CrosstableUDF(op3, "subq_avg", SubPipeline(PipelinePath([sub_op_row, sub_op_table, temp, sub_op1, sub_op_sum, sub_op_cnt, sub_op_avg])))
op5 = Filter(op4, BinOp(Field('l_quantity'),'<',Field('subq_avg')))
op6 = DropColumns(op5, ['l_extendedprice'])
#op6 = AllAggregate(op5, 'avg_yearly', 'lambda v, row: v + row["l_extendedprice"]')
#op7 = ScalarComputation({'x': op6}, 'lambda x: x["avg_yearly"]/7.0')




