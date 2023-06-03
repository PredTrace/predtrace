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

/* Split: query part 1 */
select
	c_custkey,
	c_name,
	c_acctbal,
	n_name,
	c_address,
	c_phone,
	c_comment, l_linenumber, o_orderkey, l_extendedprice, l_discount
into 
    temp_table_q10
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
	and c_nationkey = n_nationkey;


/* Split query part 2 */
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
	temp_table_q10
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

/* intermediate result size: */
SELECT count(*) from temp_table_q10;
SELECT pg_size_pretty( pg_total_relation_size('temp_table_q10'));


/* ----------------- */
/* Step 3: lineage query */
/* query on inter result */
SELECT * FROM temp_table_q10 where c_custkey = 57040 and c_name = 'Customer#000057040' and c_acctbal = 632.87 and n_name = 'JAPAN' and c_address = 'Eioyzjf4pp' and c_phone = '22-895-641-3466';
	-- customer,
	-- orders,
	-- lineitem,
	-- nation

/* query on order, customer, lineitem tables */
SELECT * FROM nation where n_name = 'JAPAN'; /* 0.400 ms */
SELECT * FROM orders where o_orderkey IN (SELECT o_orderkey FROM temp_table_q10 where c_custkey = 57040 and c_name = 'Customer#000057040' and c_acctbal = 632.87 and n_name = 'JAPAN' and c_address = 'Eioyzjf4pp' and c_phone = '22-895-641-3466'); /* 16.192 ms */
SELECT * from customer where c_custkey = 57040;/*  0.373 ms */ 
SELECT * from lineitem where l_linenumber IN (SELECT l_linenumber FROM temp_table_q10 where c_custkey = 57040 and c_name = 'Customer#000057040' and c_acctbal = 632.87 and n_name = 'JAPAN' and c_address = 'Eioyzjf4pp' and c_phone = '22-895-641-3466') and l_orderkey IN (SELECT l_orderkey FROM temp_table_q10 where c_custkey = 57040 and c_name = 'Customer#000057040' and c_acctbal = 632.87 and n_name = 'JAPAN' and c_address = 'Eioyzjf4pp' and c_phone = '22-895-641-3466');/* 15.939 ms */ 


/* finally drop the snapshot if needed */
drop table temp_table_q10;



