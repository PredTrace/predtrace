import sys
sys.path.append("../../../")
from interface import *

"""
select
  p_brand,
  p_type,
  p_size,
  count(distinct ps_suppkey) as supplier_cnt
from
  partsupp,
  part
where
  p_partkey = ps_partkey
  and p_brand <> 'Brand#45' (b)
  and p_type not like 'MEDIUM POLISHED%' - (i)
  and p_size in (49, 14, 23, 45, 19, 3, 36, 9) (b)
  and ps_suppkey not in (
    select
      s_suppkey
    from
      supplier
    where
      s_comment like '%Customer%Complaints%'
  ) (not supported ignore)
group by
  p_brand,
  p_type,
  p_size
order by
  supplier_cnt desc,
  p_brand,
  p_type,
  p_size
"""

op1 = InitTable('data/partsupp.csv')
op2 = InitTable('data/part.csv')
op3 = InnerJoin(op1, op2, ["ps_partkey"],["p_partkey"])
#op4 = DropDuplicate(op3, ["ps_partkey"])

op4 = Filter(op3, BinOp(Field('p_brand'), '!=', Constant('Brand#45')))
op5 = Filter(op4, BinOp(Field('p_size'), 'in', Constant([49, 14, 23, 45, 19, 3, 36, 9])))


op6 = GroupBy(op5, ['p_brand','p_type', 'p_size'], \
    {
      'ps_suppkey':(Value(0),'count')}, 
	 {'ps_suppkey':'supplier_cnt'})
op7 = SortValues(op6, ['supplier_cnt','p_brand','p_type','p_size'])





