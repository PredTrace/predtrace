
/* Trace */
Select * from customer where c_custkey in (
Select
    c_custkey
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
    and c_custkey = 57040 and c_name = 'Customer#000057040' and c_acctbal = 632.87 and n_name = 'JAPAN' and c_address = 'Eioyzjf4pp' and c_phone = '22-895-641-3466');


Select * from lineitem l inner join (
Select
    l_orderkey,l_linenumber
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
    and c_custkey = 57040 and c_name = 'Customer#000057040' and c_acctbal = 632.87 and n_name = 'JAPAN' and c_address = 'Eioyzjf4pp' and c_phone = '22-895-641-3466') x
On l.l_orderkey=x.l_orderkey and l.l_linenumber=x.l_linenumber;



/* Panda */
select
	c_custkey,
	c_name,
	c_acctbal,
	n_name,
	c_address,
	c_phone,
	c_comment, l_orderkey, l_extendedprice, l_discount
into 
    aug10
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


select
	c_custkey,
	c_name,
	c_acctbal,
	n_name,
	c_address,
	c_phone,
	c_comment, l_extendedprice, l_discount
into 
    view10
from
	aug10;

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
	view10
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

select * from customer where c_custkey= 57040;

SELECT * FROM nation where n_name = 'JAPAN';

select * from orders where  o_orderdate >= date '1993-10-01' and o_orderdate < date '1993-10-01' + interval '3' month and o_custkey = 57040
and o_orderkey in (select l_orderkey from aug10 where c_custkey = 57040 and c_name = 'Customer#000057040' and c_acctbal = 632.87 and n_name = 'JAPAN' and c_address = 'Eioyzjf4pp' and c_phone = '22-895-641-3466');

select * from lineitem where l_returnflag = 'R' and l_orderkey in (select l_orderkey from aug10 where c_custkey = 57040 and c_name = 'Customer#000057040' and c_acctbal = 632.87 and n_name = 'JAPAN' and c_address = 'Eioyzjf4pp' and c_phone = '22-895-641-3466');

SELECT pg_size_pretty( pg_total_relation_size('aug10'));

/* PERM use docker image */