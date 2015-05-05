# Before running this script you should set the enviroment variable HADOOP_HOME (e.g. export HADOOP_HOME=/usr/lib/hadoop-2.4.0)
# Usage:
#  hadoop jar target/spca-1.0-SNAPSHOT-job.jar \
#  ... # other options
#  -i=<path/to/input/matrix/on/hdfs> -o=<path/to/outputfolder/on/hdfs> -rows=<number of rows> -cols=<number of columns> -pcs=<number of principal components> [-DerrSampleRate=<Error sampling rate>] [-DmaxIter=<max iterations>] [-normalize<0/1 (normalize input matrix or not)>] 



# Description of arguments:
#       <path/to/input/matrix/on/hdfs>: Hdfs directory that contains an example input matrix in the sequenceFileFormat <IntWritable key, VectorWritable value>.
#       <path/to/outputfolder/on/hdfs>: Hdfs directory where the resulting principal components is written
#       <number of rows>: Number of rows for the input matrix
#       <number of columns>: Number of columns for the input matrix
#       <number of principal components>: Number of desired principal components
#       [<Error sampling rate>](optional): The error sampling rate [0-1] that is used for computing the error, It can be set to 0.01 to compute the error for only a small sample of the matrix, this speeds up the computations significantly)
#       [<max iterations>] (optional): Maximum number of iterations before terminating, the default is 3
#	<0/1 (normalize input matrix or not)>](optional) : 0 or 1 values that specifies whether the input matrix needs to be normalized or not. 1 means that the matrix should be normalized, 0 means that matrix should not be normalized. Normalization is done by dividing each column by (column_max-column_min)

LOG=example.log
$HADOOP_HOME/bin/hadoop jar target/spca-1.0-SNAPSHOT-job.jar org.qcri.pca.SPCADriver -D mapred.cluster.map.memory.mb=3072 -D mapred.job.map.memory.mb=3072 -D mapreduce.map.memory.mb=3072 -D mapred.task.timeout=6000000 -D mapred.healthChecker.script.timeout=30000 -D mapred.job.reuse.jvm.num.tasks=1  -i seqfiles -o output --tempDir tmp  -rows 7 -cols 5 -pcs 3 -errSampleRate 1 2>&1 | tee $LOG