import sys
sys.path.append("../../../")
from interface import *

"""
select
	s_acctbal,
	s_name,
	n_name,
	p_partkey,
	p_mfgr,
	s_address,
	s_phone,
	s_comment
from
	part,
	supplier,
	partsupp,
	nation,
	region
where
	p_partkey = ps_partkey
	and s_suppkey = ps_suppkey
	and p_size = :1
	and p_type like '%:2'
	and s_nationkey = n_nationkey
	and n_regionkey = r_regionkey
	and r_name = ':3'
	and ps_supplycost = (
		select
			min(ps_supplycost)
		from
			partsupp,
			supplier,
			nation,
			region
		where
			p_partkey = ps_partkey
			and s_suppkey = ps_suppkey
			and s_nationkey = n_nationkey
			and n_regionkey = r_regionkey
			and r_name = ':3'
	)
order by
	s_acctbal desc,
	n_name,
	s_name,
	p_partkey;
"""

part = InitTable('data/part.csv')
supplier = InitTable('data/supplier.csv')
partsupp = InitTable('data/partsupp.csv')
nation = InitTable('data/nation.csv')
region = InitTable('data/region.csv')
op1 = InnerJoin(region, nation, 'n_regionkey','r_regionkey')
op2 = InnerJoin(op1, supplier, ['s_nationkey'],['n_nationkey'])
op3 = InnerJoin(op2, partsupp,  ['ps_suppkey'],['s_suppkey'])
op4 = InnerJoin(part, op3, ['ps_partkey'],['p_partkey'])

op5 = Filter(op4, And(BinOp(Field('p_size'),'==',Constant(15)), BinOp(Constant('Brand'), 'subset', Field('p_type'))))
op6 = Filter(op5, BinOp(Field('r_name'),'==',Constant('EUROPE')))

sub_op_row = SubpipeInput(op6, 'row')
sub_op_table = SubpipeInput(partsupp, 'table')
op2_1 = InnerJoin(sub_op_table, supplier, ['ps_suppkey'],['s_suppkey'])
op3_1 = InnerJoin(op2_1, nation, ['s_nationkey'],['n_nationkey'])
op4_2 = InnerJoin(op3_1, region, ['n_regionkey'],['r_regionkey'])
temp1 = ScalarComputation({'row':sub_op_row}, 'lambda row: row["p_partkey"]')
op5_1 = Filter(op4_2, BinOp(Field('ps_partkey'), '==', temp1))
op5_2 = Filter(op5_1, BinOp(Field('r_name'),'==',Constant('EUROPE')))
op6_2 = AllAggregate(op5_2, Value(0), 'lambda v,row: v if v < row["ps_supplycost"] else row["ps_supplycost"]') # min

op7 = CrosstableUDF(op6, "subq_min", SubPipeline(PipelinePath([sub_op_row, sub_op_table, op2_1, op3_1, op4_2, temp1, op5_1, op5_2, op6_2])))

op8 = Filter(op7, BinOp(Field('ps_supplycost'), '==', Field('subq_min')))
op9 = SortValues(op8, ['s_acctbal','n_name','s_name','p_partkey'])
op10 = DropColumns(op9, ['s_acctbal','s_name','n_name','p_partkey','p_mfgr','s_address','s_phone','s_comment'])