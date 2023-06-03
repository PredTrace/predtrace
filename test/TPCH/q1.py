import sys
sys.path.append("../../../")
from interface import *

"""
select
	l_returnflag,
	l_linestatus,
	sum(l_quantity) as sum_qty,
	sum(l_extendedprice) as sum_base_price,
	sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
	sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
	avg(l_quantity) as avg_qty,
	avg(l_extendedprice) as avg_price,
	avg(l_discount) as avg_disc,
	count(*) as count_order
from
	lineitem
where
	l_shipdate <= date '1998-12-01' - interval ':1' day
group by
	l_returnflag,
	l_linestatus
order by
	l_returnflag,
	l_linestatus
LIMIT 1;
"""

op1 = InitTable('data/lineitem.csv')
op2 = Filter(op1, BinOp(Field('l_shipdate'),'<=',Constant('1998-11-31')))
op3 = GroupBy(op2, ['l_returnflag','l_linestatus'], \
    {'l_quantity':(Value(0), 'sum'),\
     'l_extendedprice':(Value(0), 'sum'),\
     'sum_disc_price':(Value(0), 'lambda row: row["l_extendedprice"]*(1-row["l_discount"])'),\
     'sum_charge':(Value(0), 'lambda row: row["l_extendedprice"]*(1-row["l_discount"])*(1+row["l_tax"])'),\
	 'l_quantity':(Value(0),'avg'),\
	 'l_extendedprice':(Value(0),'avg'),
     'l_discount':(Value(0),'avg'),\
     'count_order':(Value(0),'count')}, {'l_quantity':'sum_qty','l_extendedprice':'sum_base_price','sum_disc_price':'sum_disc_price','sum_charge':'sum_charge',\
	 'l_quantity':'avg_qty','l_extendedprice':'avg_price','l_discount':'avg_disc','count_order':'count_order'})
op4 = TopN(op3, 1, ['l_returnflag','l_linestatus'])
