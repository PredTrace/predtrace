

 # Efficient Row-Level Lineage Leveraging Predicate Pushdown

 Row-level lineage explains what input rows produce an output row through a workflow, having many applications like data debugging, auditing, data integration, etc. Prior work on lineage falls in two lines: eager lineage tracking and lazy lineage inference. Eager tracking integrates lineage tracing tightly into the operator implementation, enabling efficient customized tracking. However, this approach is intrusive, system-specific, and lacks adaptability. In contrast, lazy inference generates additional queries to compute lineage; it can be easily applied to any database, but the lineage query is usually slow. Furthermore, both approaches have limited coverage of the type of query/workflow supported due to operator-specific tracking or inference rules.

We propose PredTrace, a different approach that achieves easy adaptation, low runtime overhead, efficient lineage querying, and high query/workflow coverage. It leverages predicate pushdown to infer data lineage, building on top of MagicPush, a powerful predicate pushdown mechanism that has great coverage of query types (including correlated subqueries) and operators (including many non-relational operators and UDFs). 

## Prerequisites
To install the package and prepare for use, run:
<pre><code>git clone https://github.com/PredTrace/predtrace.git

pip install -r requirements.txt
</code></pre>

The following python packages are required to run PredTrace and z3.

## Demo
We will update the demo soon.

## Datasets
TPC-H: https://www.tpc.org/tpch/.

Data Science pipeline: https://github.com/congy/AutoSuggest.
