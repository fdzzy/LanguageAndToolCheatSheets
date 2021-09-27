
## Example
Assume we have data
```
001,Rajiv,Hyderabad
002,siddarth,Kolkata
003,Rajesh,Delhi
```
We can have a script `sample_script.pig`:
```sql
student = LOAD 'hdfs://localhost:9000/pig_data/student.txt' USING
   PigStorage(',') as (id:int,name:chararray,city:chararray);
```
In grunt shell we run:
```bash
grunt> run /sample_script.pig
grunt> Dump;

(1,Rajiv,Hyderabad)
(2,siddarth,Kolkata)
(3,Rajesh,Delhi)
```

### Data types
```
int, long, float, double, chararray: represent a string in UTF-8 format, Bytearray: represent a blob,
Boolean, Datetime, Biginteger, Bigdecimal
Complex Types: Tuple, Bag: a bag is a collection of tuples, Map
```

### Load Operator
```sql
Relation_name = LOAD 'Input file path' USING function as schema;
# function − We have to choose a function from the set of load functions provided by Apache Pig (BinStorage, JsonLoader, PigStorage, TextLoader).
# schema - (column1 : data type, column2 : data type, column3 : data type);
```

### Store Operator
```sql
STORE Relation_name INTO 'required_directory_path' [USING function];
```

### Describe
The describe operator is used to view the schema of a relation.
```sql
describe student;
```
output
```
student: { id: int,firstname: chararray,lastname: chararray,phone: chararray,city: chararray }
```

### Explain
The explain operator is used to display the logical, physical, and MapReduce execution plans of a relation.
```sql
explain Relation_name;
```

### Illustrate
The illustrate operator gives you the step-by-step execution of a sequence of statements.
```sql
illustrate Relation_name;
```
Example output
```
INFO  org.apache.pig.backend.hadoop.executionengine.mapReduceLayer.PigMapOnly$M ap - Aliases
being processed per job phase (AliasName[line,offset]): M: student[1,10] C:  R:
---------------------------------------------------------------------------------------------
|student | id:int | firstname:chararray | lastname:chararray | phone:chararray | city:chararray |
--------------------------------------------------------------------------------------------- 
|        | 002    | siddarth            | Battacharya        | 9848022338      | Kolkata        |
---------------------------------------------------------------------------------------------
```

### GROUP
The GROUP operator is used to group the data in one or more relations. It collects the data having the same key.
```sql
Group_data = GROUP Relation_name BY key;
# example
group_data = GROUP student_details by age;
Dump group_data;
```
Output: the resulting schema has two columns:
- One is age, by which we have grouped the relation.
- The other is a bag, which contains the group of tuples, student records with the respective age.
```sql
Describe group_data;
```
output:
```
group_data: {group: int,student_details: {(id: int,firstname: chararray,
               lastname: chararray,age: int,phone: chararray,city: chararray)}}
```
Group by multiple columns
```sql
group_multiple = GROUP student_details by (age, city);
Dump group_multiple; 
```
Output
```
((21,Pune),{(4,Preethi,Agarwal,21,9848022330,Pune)})
((21,Hyderabad),{(1,Rajiv,Reddy,21,9848022337,Hyderabad)})
((22,Delhi),{(3,Rajesh,Khanna,22,9848022339,Delhi)})
((22,Kolkata),{(2,siddarth,Battacharya,22,9848022338,Kolkata)})
((23,Chennai),{(6,Archana,Mishra,23,9848022335,Chennai)})
((23,Bhuwaneshwar),{(5,Trupthi,Mohanthy,23,9848022336,Bhuwaneshwar)})
((24,Chennai),{(8,Bharathi,Nambiayar,24,9848022333,Chennai)})
(24,trivendram),{(7,Komal,Nayak,24,9848022334,trivendram)})
```

### JOIN
Self join
```sql
customers1 = LOAD 'hdfs://localhost:9000/pig_data/customers.txt' USING PigStorage(',')
   as (id:int, name:chararray, age:int, address:chararray, salary:int);
  
customers2 = LOAD 'hdfs://localhost:9000/pig_data/customers.txt' USING PigStorage(',')
   as (id:int, name:chararray, age:int, address:chararray, salary:int); 

# Relation3_name = JOIN Relation1_name BY key, Relation2_name BY key ;
customers3 = JOIN customers1 BY id, customers2 BY id;
```
Inner join
```sql
customers = LOAD 'hdfs://localhost:9000/pig_data/customers.txt' USING PigStorage(',')
   as (id:int, name:chararray, age:int, address:chararray, salary:int);
orders = LOAD 'hdfs://localhost:9000/pig_data/orders.txt' USING PigStorage(',')
   as (oid:int, date:chararray, customer_id:int, amount:int);
coustomer_orders = JOIN customers BY id, orders BY customer_id;
```
Outer join
```sql
# left outer join
outer_left = JOIN customers BY id LEFT OUTER, orders BY customer_id;
# right outer join
outer_right = JOIN customers BY id RIGHT, orders BY customer_id;
# full outer join
outer_full = JOIN customers BY id FULL OUTER, orders BY customer_id;
```
Using multiple keys
```sql
Relation3_name = JOIN Relation2_name BY (key1, key2), Relation3_name BY (key1, key2);
```
Example
```
QueryCounts = load 'data_path' using PigStorage('\t') as (query:chararray, lang:chararray, query_count:long);
B = foreach A generate query, lang, type, typ_q, title, match_type, vc, mvc, (STARTSWITH(category, 'human/') ? 'general_people' : category) as category2;

C1 = JOIN B BY (query, lang), QueryCounts BY (query, lang);
C = foreach C1 generate B::query as query, B::lang as lang, QueryCounts::query_count as query_count, B::category2 as category2;
```

### CROSS
The CROSS operator computes the cross-product of two or more relations. This chapter explains with example how to use the cross operator in Pig Latin.
```sql
Relation3_name = CROSS Relation1_name, Relation2_name;
```

### UNION
The UNION operator of Pig Latin is used to merge the content of two relations. To perform UNION operation on two relations, their columns and domains must be identical.
```sql
Relation_name3 = UNION Relation_name1, Relation_name2;
```

### SPLIT
The SPLIT operator is used to split a relation into two or more relations.
```sql
SPLIT Relation1_name INTO Relation2_name IF (condition1), Relation2_name (condition2);

student_details = LOAD 'hdfs://localhost:9000/pig_data/student_details.txt' USING PigStorage(',')
   as (id:int, firstname:chararray, lastname:chararray, age:int, phone:chararray, city:chararray);
SPLIT student_details into student_details1 if age<23, student_details2 if (22<age and age>25);
```

### FILTER
The FILTER operator is used to select the required tuples from a relation based on a condition.
```sql
Relation2_name = FILTER Relation1_name BY (condition);

filter_data = FILTER student_details BY city == 'Chennai';
```

### DISTINCT
The DISTINCT operator is used to remove redundant (duplicate) tuples from a relation.
```sql
Relation_name2 = DISTINCT Relatin_name1;
```

### FOREACH
The FOREACH operator is used to generate specified data transformations based on the column data.
```sql
Relation_name2 = FOREACH Relatin_name1 GENERATE (required data);

student_details = LOAD 'hdfs://localhost:9000/pig_data/student_details.txt' USING PigStorage(',')
   as (id:int, firstname:chararray, lastname:chararray,age:int, phone:chararray, city:chararray);
foreach_data = FOREACH student_details GENERATE id,age,city;
```

### ORDER BY
The ORDER BY operator is used to display the contents of a relation in a sorted order based on one or more fields.
```sql
Relation_name2 = ORDER Relatin_name1 BY (ASC|DESC);

order_by_data = ORDER student_details BY age DESC;
```

### LIMIT
The LIMIT operator is used to get a limited number of tuples from a relation.
```sql
Result = LIMIT Relation_name <required number of tuples>;

limit_data = LIMIT student_details 4; 
```
Group by, then limit
```sql
B1 = foreach (group A by (query, info)) generate flatten(group) as (query, lang), SUM(A9.traffic) as count;
B = filter B1 by count >= 10;

C = foreach (group B by lang) {
    C1 = order B by count DESC;
    C2 = limit C1 100;
    generate flatten(C2);
}
```

### SUM
```sql
B1 = foreach (group A by (query, info)) generate flatten(group) as (query, lang), SUM(A9.traffic) as count;
```