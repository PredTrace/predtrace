import sys
sys.path.append("../../")
from interface import *

"""
select
  s_name,
  s_address
from
  supplier, nation
where
  s_suppkey in (
    select
      ps_suppkey
    from
      partsupp
    where
      ps_partkey in (
        select
          p_partkey
        from
          part
        where
          p_name like 'forest%'
        ) 
      and ps_availqty > (
        select
          0.5 * sum(l_quantity)
        from
          lineitem
        where
          l_partkey = ps_partkey
          and l_suppkey = ps_suppkey
          and l_shipdate >= '1994-01-01'
          and l_shipdate < '1995-01-01'
        )
    )
  and s_nationkey = n_nationkey
  and n_name = 'CANADA'
order by
  s_name
"""


lineitem = InitTable('data/lineitem.csv')
part = InitTable('data/part.csv')
partsupp = InitTable('data/partsupp.csv')
supplier = InitTable('data/supplier.csv')
nation = InitTable('data/nation.csv')


sub_op_row = SubpipeInput(partsupp, 'row')
sub_op_table = SubpipeInput(part, 'table')
sub_op1 = Filter(sub_op_table, BinOp(Constant('forest'), 'subset', Field('p_name')))
temp1 = ScalarComputation({'x':sub_op_row},'lambda x: x["ps_partkey"]')
sub_op2 = Filter(sub_op1, BinOp(Field('p_partkey'), '==', temp1))
sub_op3 = AllAggregate(sub_op2, Value(0), 'lambda v,row: 1')
nested_op1 = CrosstableUDF(partsupp, 'nested_q1', SubPipeline(PipelinePath([sub_op_row, sub_op_table, sub_op1, temp1, sub_op2, sub_op3])))
op1 = Filter(nested_op1, BinOp(Field('nested_q1'), '==', Constant(1)))

sub_op_row = SubpipeInput(op1, 'row')
sub_op_table = SubpipeInput(lineitem, 'table')
sub_op1 = Filter(sub_op_table, And(BinOp(Field('l_shipdate'),'subset',Constant('1994-01-01')), BinOp(Field('l_shipdate'),'subset',Constant('1995-01-01'))))
temp2 = ScalarComputation({'x':sub_op_row},'lambda x: x["ps_partkey"]')
temp3 = ScalarComputation({'x':sub_op_row},'lambda x: x["ps_suppkey"]')
sub_op2 = Filter(sub_op1, And(BinOp(Field('l_partkey'),'==',temp2), BinOp(Field('l_suppkey'),'==',temp3)))
sub_op3 = AllAggregate(sub_op2, Value(0), 'lambda v,row: 0.5*row["l_quantity"]')
nested_op2 = CrosstableUDF(op1,'nested_q2', SubPipeline(PipelinePath([sub_op_row, sub_op_table, sub_op1, temp2, temp3, sub_op2, sub_op3])))
op2 = Filter(nested_op2, BinOp(Field('ps_availqty'), '>', Field('nested_q2')))

sub_op_row = SubpipeInput(supplier, 'row')
sub_op_table = SubpipeInput(op2, 'table')
temp4 = ScalarComputation({'x':sub_op_row}, 'lambda x: x["s_suppkey"]')
sub_op1 = Filter(sub_op_table, BinOp(Field('ps_suppkey'),'==',temp4))
sub_op2 = AllAggregate(sub_op1, Value(0), 'lambda v,row: 1')
out_nested = CrosstableUDF(supplier, 'out_nested', SubPipeline(PipelinePath([sub_op_row, sub_op_table, temp4, sub_op1, sub_op2])))
op3 = Filter(out_nested, BinOp(Field('out_nested'),'==',Constant(1)))

op4 = InnerJoin(op3, nation, ['s_nationkey'], ['n_nationkey'])
# not minimal, take snapshot here, should including s_nationkey
op5 = Filter(op4, BinOp(Field('n_name'), '==', Constant('CANADA')))
op6 = SortValues(op5, ['s_name'])
op7 = DropColumns(op6, ['s_name','s_address'])
# s_name = v




