
```bash
# merge a folder of part files into one
$ hadoop fs -cat hdfs://path/to/folder/* >data.txt
$ hdfs dfs -cat hdfs://path/to/folder/* >data.txt
# copy local file to hdfs
$ hdfs dfs -put qid_list_all.txt hdfs://path/to/folder/
```