import sys
sys.path.append("../../../")
from interface import *

"""
select
  c_name,
  c_custkey,
  o_orderkey,
  o_orderdate,
  o_totalprice,
  sum(l_quantity)
from
  customer,
  orders,
  lineitem
where
  o_orderkey in (
    select
      l_orderkey
    from
      lineitem
    group by
      l_orderkey
    having
      sum(l_quantity) > 300
    )  Yin: this operation is not supported
  and c_custkey = o_custkey
  and o_orderkey = l_orderkey
group by
  c_name,
  c_custkey,
  o_orderkey,
  o_orderdate,
  o_totalprice
order by
  o_totalprice desc,
  o_orderdate,
  o_orderkey
limit 100
"""


customer = InitTable('data/customer.csv')
orders = InitTable('data/orders.csv')
lineitem = InitTable('data/lineitem.csv')
op1 = InnerJoin(customer, orders, ["c_custkey"],["o_custkey"])
op2 = InnerJoin(op1, lineitem, ["o_orderkey"],["l_orderkey"])

sub_op_row = SubpipeInput(op2, 'row')
sub_op_table = SubpipeInput(lineitem, 'table')
temp1 = ScalarComputation({'row':sub_op_row}, 'lambda row: row["o_orderkey"]')
op_1 = GroupBy(sub_op_table, ['l_orderkey'], {
      'l_quantity':(Value(0),'sum')}, 
	 {'l_quantity':'sum_quantity'})
op_2 = Filter(op_1, BinOp(Field('sum_quantity'),'>',Constant(100)))
op_4 = Filter(op_2, BinOp(Field('l_orderkey'), '==', temp1))
op_5 = AllAggregate(op_4, Value(0), 'lambda v,row: v+1') # count

op3 = CrosstableUDF(op2, "orderkey_match", SubPipeline(PipelinePath([sub_op_row, sub_op_table, temp1, op_1,op_2,op_4,op_5])))


op4 = GroupBy(op3, ['c_name','c_custkey', 'o_orderkey', 'o_orderdate', 'o_totalprice'], \
    {
      'l_quantity':(Value(0),'sum')}, 
	 {'l_quantity':'sum_quantity'})
op5 = TopN(op4, 100, ["o_totalprice","o_orderdate","o_orderkey"])


