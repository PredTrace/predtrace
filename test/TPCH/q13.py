import sys
sys.path.append("../../../")
from interface import *


"""
select
  c_count,
  count(*) as custdist
from (
  select
    c_custkey,
    count(o_orderkey) as c_count
  from
    customer left outer join orders on (
      c_custkey = o_custkey
      and o_comment not like '%special%requests%'
    )
  group by
    c_custkey
  ) as c_orders
group by
  c_count
order by
  custdist desc,
  c_count desc
"""


op1 = InitTable('data/orders.csv')
op2 = InitTable('data/customer.csv')
op3 = LeftOuterJoin(op1, op2,['c_custkey'],['o_custkey'])
op4 = GroupBy(op3, ['c_custkey'], \
    {
     'o_orderkey':(Value(0), 'count')}, 
	 {'o_orderkey':'c_count'})
op5 = GroupBy(op4, ['c_count'], \
    {
     'custdist':(Value(0), 'count')}, 
	 {'custdist':'custdist'})
op6 = SortValues(op5, ['custdist', 'c_count'])
