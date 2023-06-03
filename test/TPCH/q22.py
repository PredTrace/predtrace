import sys
sys.path.append("../../../")
from interface import *

"""
select
  cntrycode,
  count(*) as numcust,
  sum(c_acctbal) as totacctbal


from (
  select
    substr(c_phone, 1, 2) as cntrycode,
    c_acctbal
  from
    customer
  where
    substr(c_phone, 1, 2) in ('13', '31', '23', '29', '30', '18', '17')
    and c_acctbal > (


      select
        avg(c_acctbal)
      from
        customer
      where
        c_acctbal > 0.00
        and substr(c_phone, 1, 2) in ('13', '31', '23', '29', '30', '18', '17')


      )
    and not exists (


      select
        *
      from
        orders
      where
        o_custkey = c_custkey
    )


  ) as custsale



group by
  cntrycode
order by
  cntrycode
"""

customer = InitTable('data/customer.csv')
op1 = SetItem(customer, 'cntrycode', 'lambda row: substr(row["c_phone"], 1, 2)')
op2 = Filter(op1, BinOp(Field('cntrycode'), 'in', Constant(['13', '31', '23', '29', '30', '18', '17'])))

sub1_op_row = SubpipeInput(op2, 'row')
sub1_op_table = SubpipeInput(customer, 'table')
sub1_op1 = Filter(sub1_op_table, BinOp(Field('c_acctbal'), '>', Constant(0)))
sub1_op2 = SetItem(sub1_op1, 'cntrycode', 'lambda row: substr(row["c_phone"], 1, 2)')
sub1_op3 = Filter(sub1_op2, BinOp(Field('cntrycode'), 'in', Constant(['13', '31', '23', '29', '30', '18', '17'])))
sub1_sum = AllAggregate(sub1_op3, Value(0), 'lambda v,row: v + row["c_acctbal"]')
sub1_cnt = AllAggregate(sub1_op3, Value(0), 'lambda v,row: v + 1')
sub1_avg = ScalarComputation({'s':sub1_sum, 'c':sub1_cnt},'lambda s, c: s/c')

op3 = CrosstableUDF(op2, "subq_avg", SubPipeline(PipelinePath([sub1_op_row, sub1_op_table, sub1_op1, sub1_op2, sub1_op3,sub1_sum, sub1_cnt, sub1_avg])))
op4 = Filter(op3, BinOp(Field('c_acctbal'),'>', Field('subq_avg')))


orders = InitTable('data/orders.csv')
sub2_op_row = SubpipeInput(op4, 'row')
sub2_op_table = SubpipeInput(orders, 'table')
sub2_op1 = ScalarComputation({'row':sub2_op_row}, 'lambda row: row["c_custkey"]')
sub2_op2 = Filter(sub2_op_table, BinOp(Field('o_custkey'),'==',sub2_op1))
sub2_op3 = AllAggregate(sub2_op2, Value(0), 'lambda v,row: v+1')
op5 = CrosstableUDF(op4, "subq_notexist", SubPipeline(PipelinePath([sub2_op_row, sub2_op_table, sub2_op1, sub2_op2, sub2_op3])))
op6 = Filter(op5, BinOp(Field('subq_notexist'),'<=', Constant(0)))


op7 = GroupBy(op6, ['cntrycode'], \
    {
      'numcust':(Value(0),'count'),\
      'c_acctbal':(Value(0),'sum')}, \
	 {'numwait':'numwait', 'c_acctbal':'totacctbal'})

op8 = SortValues(op7, ['cntrycode'])



