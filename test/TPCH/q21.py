import sys
sys.path.append("../../../")
from interface import *

"""
select
  s_name,
  count(*) as numwait
from
  supplier,
  lineitem l1,
  orders,
  nation
where
  s_suppkey = l1.l_suppkey k
  and o_orderkey = l1.l_orderkey k
  and o_orderstatus = 'F'
  and l1.l_receiptdate > l1.l_commitdate
  and exists (
    select
      *
    from
      lineitem l2
    where
      l2.l_orderkey = l1.l_orderkey
      and l2.l_suppkey <> l1.l_suppkey
  )
  and not exists (
    select
      *
    from
      lineitem l3
    where
      l3.l_orderkey = l1.l_orderkey k 
      and l3.l_suppkey <> l1.l_suppkey
      and l3.l_receiptdate > l3.l_commitdate
  )
  and s_nationkey = n_nationkey
  and n_name = 'SAUDI ARABIA'
group by
  s_name
order by
  numwait desc,
  s_name
limit 100
"""

lineitem1 = InitTable('data/lineitem.csv')
lineitem2 = InitTable('data/lineitem.csv')
lineitem3 = InitTable('data/lineitem.csv')
supplier = InitTable('data/supplier.csv')
orders = InitTable('data/orders.csv')
nation = InitTable('data/nation.csv')
op1 = InnerJoin(supplier, lineitem1, ["s_suppkey"],["l_suppkey"])
op2 = InnerJoin(op1, orders, ["l_orderkey"],["o_orderkey"])
op3 = InnerJoin(op2, nation, ["s_nationkey"], ["n_nationkey"])
op4 = Filter(op3, BinOp(Field('o_orderstatus'), '==', Constant('F')))
op5 = Filter(op4, BinOp(Field('l_receiptdate'), 'subset', Field('l_commitdate')))
op6 = Filter(op5, BinOp(Field('n_name'), '==', Constant('SAUDI ARABIA')))

# exist condition 
sub1_op_row = SubpipeInput(op6, 'row')
sub1_op_table = SubpipeInput(lineitem2, 'table')
#sub1_op1 = InnerJoin(op6, lineitem2, ["l_orderkey"],["l_orderkey"])
temp1 = ScalarComputation({'x':sub1_op_row}, 'lambda x: x["l_orderkey"]')
sub1_op1 = Filter(sub1_op_table, BinOp(Field('l_orderkey'),'==',temp1))
temp2 = ScalarComputation({'x':sub1_op_row}, 'lambda x: x["l_suppkey"]')
sub1_op2 = Filter(sub1_op1, BinOp(Field('l_suppkey'), '!=', temp2))
#sub1_op3 = ScalarComputation({'count':sub1_op2}, 'lambda count: count>0')
sub1_op3 = AllAggregate(sub1_op2, Value(0), 'lambda count,row: count+1')
op7 = CrosstableUDF(op6, "subq_exist", SubPipeline(PipelinePath([sub1_op_row, sub1_op_table, temp1, sub1_op1, temp2, sub1_op2, sub1_op3])))
op8 = Filter(op7, BinOp(Field('subq_exist'),'>', Constant(0)))

# not exist condition

sub2_op_row = SubpipeInput(op8, 'row')
sub2_op_table = SubpipeInput(lineitem3, 'table')
#sub2_op1 = InnerJoin(op8, lineitem3, ["l_orderkey"],["l_orderkey"])
temp3 = ScalarComputation({'x':sub2_op_row}, 'lambda x: x["l_orderkey"]')
sub2_op1 = Filter(sub2_op_table, BinOp(Field('l_orderkey'),'==',temp2))
temp4 = ScalarComputation({'x':sub2_op_row}, 'lambda x: x["l_suppkey"]')
temp5 = ScalarComputation({'x':sub2_op_row}, 'lambda x: x["l_receiptdate"]')
sub2_op2 = Filter(sub2_op1, BinOp(Field('l_suppkey'), '!=', temp4))
sub2_op3 = Filter(sub2_op2, BinOp(Field('l_receiptdate'), 'subset', temp5))
#sub2_op4 = ScalarComputation({'count':sub2_op3}, 'lambda count: count>0')
sub2_op4 = AllAggregate(sub2_op3, Value(0), 'lambda count,row: count+1')
op9 = CrosstableUDF(op8, "subq_notexist", SubPipeline(PipelinePath([sub2_op_row, sub2_op_table, temp3, sub2_op1, temp4, temp5, sub2_op2, sub2_op3, sub2_op4])))
op10 = Filter(op9, BinOp(Field('subq_notexist'),'<=', Constant(0)))



op11 = GroupBy(op10, ['s_name'], \
    {
      'numwait':(Value(0),'count')}, 
	 {'numwait':'numwait'})
op12 = TopN(op11, 100, ["numwait","s_name"])







