import sys
sys.path.append("../../../")
from interface import *


"""
select
  *
from (
  select
    ps_partkey,
    sum(ps_supplycost * ps_availqty) as value
  from
    partsupp,
    supplier,
    nation
  where
    ps_suppkey = s_suppkey
    and s_nationkey = n_nationkey
    and n_name = 'GERMANY'
  group by
    ps_partkey
) as inner_query
where
  value > (
    select
      sum(ps_supplycost * ps_availqty) * 0.0001
    from
      partsupp,
      supplier,
      nation
    where
      ps_suppkey = s_suppkey
      and s_nationkey = n_nationkey
      and n_name = 'GERMANY'
  )
order by
  value desc
"""



op1 = InitTable('data/nation.csv')
op3 = InitTable('data/partsupp.csv')
op4 = InitTable('data/supplier.csv')
op5 = InnerJoin(op3, op4, ["ps_suppkey"],["s_suppkey"])
op6 = InnerJoin(op5, op1, ["s_nationkey"],["n_nationkey"])
op2 = Filter(op6, BinOp(Field('n_name'), '==', Field('GERMANY')))
inner1 = GroupBy(op2, ['ps_partkey'], \
    {
      'value':(Value(0),'lambda row: row["ps_supplycost"]*row["ps_availqty"]')}, 
	 {'value':'value'})

sub_op_row = SubpipeInput(inner1, 'row')
sub_op_table = SubpipeInput(op3, 'table')
sub1 = InnerJoin(sub_op_table, op4, ["ps_suppkey"],["s_suppkey"])
sub2 = InnerJoin(sub1, op1, ["s_nationkey"],["n_nationkey"])
sub3 = Filter(sub2, BinOp(Field('n_name'), '==', Field('GERMANY')))
inner2_sum = AllAggregate(sub3, Value(0),'lambda v, row: (v+ row["ps_supplycost"]*row["ps_availqty"]) * 0.0001')
op7 = CrosstableUDF(inner1, "subq_sum", SubPipeline(PipelinePath([sub_op_row, sub_op_table, sub1, sub2, sub3, inner2_sum])))
op8 = Filter(op7, BinOp(Field('value'), '>', Field('subq_sum')))
op9 = SortValues(op8, ["value"])

