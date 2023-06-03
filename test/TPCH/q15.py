import sys
sys.path.append("../../../")
from interface import *

"""
with revenue_view as (
  select
    l_suppkey as supplier_no,
    sum(l_extendedprice * (1 - l_discount)) as total_revenue
  from
    lineitem
  where
    l_shipdate >= '1996-01-01'
    and l_shipdate < '1996-04-01'
  group by
    l_suppkey)
select
  s_suppkey,
  s_name,
  s_address,
  s_phone,
  total_revenue
from
  supplier,
  revenue_view
where
  s_suppkey = supplier_no
  and total_revenue = (
    select
      max(total_revenue)
    from
      revenue_view
    )
order by
  s_suppkey
"""

# Yin: first create view
op1 = InitTable('data/lineitem.csv')
op2 = Filter(op1, BinOp(Field('l_shipdate'),'>=',Constant('1996-01-01')))
op3 = Filter(op2, BinOp(Field('l_shipdate'),'<',Constant('1996-04-01')))
view = GroupBy(op3, ['l_suppkey'], \
    {
      'total_revenue':(Value(0),'lambda row: row["l_extendedprice"] * (1 - row["l_discount"])')}, 
	 {'l_suppkey':'supplier_no', 'total_revenue':'total_revenue'})

# 
op5 = InitTable('data/supplier.csv')
op6 = InnerJoin(view, op5, ["supplier_no"], ["s_suppkey"])

sub_op_row = SubpipeInput(op6, 'row')
sub_op_table = SubpipeInput(op6, 'table')

sub_op1 = AllAggregate(sub_op_table, Value(0), 'lambda v,row: v if v > row["total_revenue"] else row["total_revenue"]') # max
op7 = CrosstableUDF(op6, "subq_max", SubPipeline(PipelinePath([sub_op_row, sub_op_table, sub_op1])))
op8 = Filter(op7, BinOp(Field('total_revenue'), '==', Field('subq_max')))

op9 = SortValues(op8, ["s_suppkey"])