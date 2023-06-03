import sys
sys.path.append("../../../")
from interface import *

"""
select
  sum(l_extendedprice * (1 - l_discount)) as revenue
from
  lineitem,
  part
where
  p_partkey = l_partkey
  and (
    (
      p_brand = 'Brand#12'
      and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
      and l_quantity >= 1 and l_quantity <= 11
      and p_size between 1 and 5
      and l_shipmode in ('AIR', 'AIR REG')
      and l_shipinstruct = 'DELIVER IN PERSON'
    )
    or
    (
      p_brand = 'Brand#23'
      and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
      and l_quantity >= 10 and l_quantity <= 20
      and p_size between 1 and 10
      and l_shipmode in ('AIR', 'AIR REG')
      and l_shipinstruct = 'DELIVER IN PERSON'
    )
    or
    (
      p_brand = 'Brand#34'
      and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
      and l_quantity >= 20 and l_quantity <= 30
      and p_size between 1 and 15
      and l_shipmode in ('AIR', 'AIR REG')
      and l_shipinstruct = 'DELIVER IN PERSON'
    )
  )
"""

lineitem = InitTable('data/lineitem.csv')
part = InitTable('data/part.csv')
op3 = InnerJoin(lineitem, part, ["l_partkey"],["p_partkey"])
op4 = Filter(op3, AllOr( \
And(And(And(And(And(And(And(BinOp(Field('p_brand'),'==',Constant('Brand#12')), \
BinOp(Field('p_container'),'in',Constant(['SM CASE','SM BOX','SM PACK','SM PKG']))),\
BinOp(Field('l_quantity'),'>=',Constant(1))),\
BinOp(Field('l_quantity'),'<=',Constant(11))),\
BinOp(Field('p_size'),'>',Constant(1))),\
BinOp(Field('p_size'),'<',Constant(5))),\
BinOp(Field('l_shipmode'),'in',Constant(['AIR','AIR REG']))),\
BinOp(Field('l_shipinstruct'),'==',Constant('DELIVER IN PERSON'))\
),\
And(And(And(And(And(And(And(BinOp(Field('p_brand'),'==',Constant('Brand#23')), \
BinOp(Field('p_container'),'in',Constant(['MED BAG','MED BOX','MED PKG','MED PACK']))),\
BinOp(Field('l_quantity'),'>=',Constant(10))),\
BinOp(Field('l_quantity'),'<=',Constant(20))),\
BinOp(Field('p_size'),'>',Constant(1))),\
BinOp(Field('p_size'),'<',Constant(10))),\
BinOp(Field('l_shipmode'),'in',Constant(['AIR','AIR REG']))),\
BinOp(Field('l_shipinstruct'),'==',Constant('DELIVER IN PERSON'))\
),\
And(And(And(And(And(And(And(BinOp(Field('p_brand'),'==',Constant('Brand#34')), \
BinOp(Field('p_container'),'in',Constant(['LG CASE','LG BOX','LG PACK','LG PKG']))),\
BinOp(Field('l_quantity'),'>=',Constant(20))),\
BinOp(Field('l_quantity'),'<=',Constant(30))),\
BinOp(Field('p_size'),'>',Constant(1))),\
BinOp(Field('p_size'),'<',Constant(15))),\
BinOp(Field('l_shipmode'),'in',Constant(['AIR','AIR REG']))),\
BinOp(Field('l_shipinstruct'),'==',Constant('DELIVER IN PERSON'))\
    )))

op6 = DropColumns(op4, ['l_extendedprice','l_discount'])
# op5 = SetItem(op4, 'revenue', 'lambda row: row["l_extendedprice"]*(1-row["l_discount"])')
# op5 = AllAggregate(op4, Value(0), 'lambda v,row: v + (row["l_extendedprice"]*(1-row["l_discount"]))')






