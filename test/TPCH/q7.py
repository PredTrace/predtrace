
from multiprocessing import Condition
import sys

sys.path.append("../../")
sys.path.append("../")
import z3
import dis
from test_helper import *
from pandas_op import *
from util import *
import random
from constraint import *
from infer_schema import *
from predicate import *
from generate_input_filters import *

"""
select
  supp_nation,
  cust_nation,
  l_year,
  sum(volume) as revenue
from (
  select
    n1.n_name as supp_nation,
    n2.n_name as cust_nation,
    year(l_shipdate) as l_year,
    l_extendedprice * (1 - l_discount) as volume
  from
    supplier,
    lineitem,
    orders,
    customer,
    nation n1,
    nation n2
  where
    s_suppkey = l_suppkey k
    and o_orderkey = l_orderkey k 
    and c_custkey = o_custkey k 
    and s_nationkey = n1.n_nationkey k
    and c_nationkey = n2.n_nationkey k 
    and (
      (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
      or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
    )
    and l_shipdate between '1995-01-01' and '1996-12-31'
  ) as shipping
group by
  supp_nation,
  cust_nation,
  l_year
order by
  supp_nation,
  cust_nation,
  l_year
"""
supplier = InitTable('data/supplier.csv')
lineitem = InitTable('data/lineitem.csv')
orders = InitTable('data/orders.csv')
customer = InitTable('data/customer.csv')
nation1 = InitTable('data/nation.csv')
nation2 = InitTable('data/nation.csv')

# n2 - customer - order
# n1 - supplier
# then join lineitem
op1 = InnerJoin(nation1, customer,'n_nationkey','c_nationkey')
op2 = InnerJoin(op1, orders, 'c_custkey','o_custkey')
op3 = InnerJoin(nation2, supplier, 'n_nationkey','s_nationkey')
op4 = InnerJoin(op3, lineitem, 's_suppkey', 'l_suppkey')
op5 = InnerJoin(op2, op4, 'o_orderkey','l_orderkey')

op6 = Filter(op5, Or(And(BinOp(Field('n_name_x'), '==',Constant('FRANCE')),BinOp(Field('n_name_y'), '==',Constant('GERMANY'))),And(BinOp(Field('n_name_x'), '==',Constant('GERMANY')),BinOp(Field('n_name_y'), '==',Constant('FRANCE')))))
op7 = Filter(op6, And(BinOp(Field('l_shipdate'), 'subset', Constant('1995-01-01')),BinOp(Field('l_shipdate'), 'subset', Constant('1996-12-31'))))
op7_1 = Rename(op7, {'n_name_x':'supp_nation','n_name_y':'cust_nation','l_shipdate':'l_year'})
op8 = GroupBy(op7_1, ['supp_nation', 'cust_nation', 'l_year'], \
    {
      'volume':(Value(0),'sum')}, 
	 {'volume':'revenue'})
op9 = SortValues(op8, ['supp_nation', 'cust_nation', 'l_year'])


