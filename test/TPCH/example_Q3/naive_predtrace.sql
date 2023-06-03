/* Original query: */
select
	c_custkey,
	c_name,
	sum(l_extendedprice * (1 - l_discount)) as revenue,
	c_acctbal,
	n_name,
	c_address,
	c_phone,
	c_comment
from
	customer,
	orders,
	lineitem,
	nation
where
	c_custkey = o_custkey
	and l_orderkey = o_orderkey
	and o_orderdate >= date '1993-10-01'
	and o_orderdate < date '1993-10-01' + interval '3' month
	and l_returnflag = 'R'
	and c_nationkey = n_nationkey
group by
	c_custkey,
	c_name,
	c_acctbal,
	c_phone,
	n_name,
	c_address,
	c_comment
order by
	revenue desc
LIMIT 20;
/* Original query running time: 403.7 ms */

/* Step 1: preprocessing */
/* split the query execution to save intermediate results */

select
	*
into temp_table_b_q10_1
from
	customer,
	nation
where
c_nationkey = n_nationkey;


select
	*
into temp_table_b_q10_2
from
	lineitem,
	orders
where l_orderkey = o_orderkey
	and o_orderdate >= date '1993-10-01'
	and o_orderdate < date '1993-10-01' + interval '3' month
	and l_returnflag = 'R';

/* Step 2: finish query running */

select
	c_custkey,
	c_name,
	sum(l_extendedprice * (1 - l_discount)) as revenue,
	c_acctbal,
	n_name,
	c_address,
	c_phone,
	c_comment
from
	temp_table_b_q10_1, temp_table_b_q10_2
where
	c_custkey = o_custkey
group by
	c_custkey,
	c_name,
	c_acctbal,
	c_phone,
	n_name,
	c_address,
	c_comment
order by
	revenue desc
LIMIT 20;



/* Snapshot temp_table_q3 size: */
SELECT count(*) from temp_table_b_q10_1;
SELECT count(*) from temp_table_b_q10_2;
SELECT pg_size_pretty( pg_total_relation_size('temp_table_b_q10_1'));
SELECT pg_size_pretty( pg_total_relation_size('temp_table_b_q10_2'));




/* ----------------- */
/* Step 3: lineage query */
/* query on snapshot */
SELECT * FROM temp_table_b_q10_2 where o_custkey = 57040;

SELECT * FROM temp_table_b_q10_1 where c_custkey = 57040 and c_name = 'Customer#000057040' and c_acctbal = 632.87 and n_name = 'JAPAN' and c_address = 'Eioyzjf4pp' and c_phone = '22-895-641-3466';
/* 15.645 ms */
	-- customer,
	-- orders,
	-- lineitem,
	-- nation

/* query on order, customer, lineitem tables */
SELECT * FROM nation where n_name = 'JAPAN'; 
SELECT * FROM orders where o_orderkey IN (SELECT * FROM temp_table_b_q10_2 where o_custkey = 57040); 
SELECT * from customer where c_custkey = 57040;
SELECT * from lineitem where l_linenumber IN (SELECT * FROM temp_table_b_q10_2 where o_custkey = 57040) and l_orderkey IN (SELECT * FROM temp_table_b_q10_2 where o_custkey = 57040);





