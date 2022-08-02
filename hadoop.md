
```bash
# merge a folder of part files into one
$ hadoop fs -cat hdfs://path/to/folder/* >data.txt
$ hdfs dfs -cat hdfs://path/to/folder/* >data.txt
# copy local file to hdfs
$ hdfs dfs -put qid_list_all.txt hdfs://path/to/folder/
# view gzip file
$ hadoop fs -cat hdfs://path/to/folder/part-000000.gz | zmore
```


```bash
hadoop fs -test -d $hdfs_dir
ret_val=$?
if [ $ret_val -eq 0 ]; then
    echo "do something"
fi

function remove_hdfs_dir() {
    hdfs_dir=$1
    
    hadoop fs -test -d $hdfs_dir
    ret_val=$?
    if [ $ret_val -eq 0 ]; then
        echo "directory exist: $hdfs_dir"
        hadoop fs -rm -r $hdfs_dir
    fi
}
```