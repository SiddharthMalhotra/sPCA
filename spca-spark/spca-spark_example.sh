# Before running this script you should set the enviroment variable SPARK_HOME (e.g. export SPARK_HOME=/path/to/spark_home_dirrectory)
# Usage:
# ./bin/spark-submit \
# --class org.qcri.sparkpca.SparkPCA
#  --master <master-url> \
#  ... # other options
#  ./target/sparkPCA-1.0.jar \
#  -Di=<path/to/input/matrix> -Do=<path/to/outputfolder> -Drows=<number of rows> -Dcols=<number of columns> -Dpcs=<number of principal components> [-DerrSampleRate=<Error sampling rate>] [-DmaxIter=<max iterations>] [-DoutFmt=<output format>] [-DComputeProjectedMatrix=<0/1 (compute projected matrix or not)>] 



# Description of arguments:
# 	<master-url>: The master URL for the cluster (e.g. spark://23.195.26.187:7077), it is set to `local[K]` for running Spark in the local mode with *K* threads (ideally, set *K* to the number of cores on your machine). If this argument is set to `local`, the applications runs locally on one worker thread (i.e., no parlellism at all).
#	<path/to/input/matrix>: File or directory that contains an input matrix in the sequenceFileFormat `<IntWritable key, VectorWritable value>`.
#	<path/to/outputfolder>: The directory where the resulting principal components is written
#	<number of rows>: Number of rows for the input matrix 
#	<number of columns>: Number of columns for the input matrix 
#	<number of principal components>: Number of desired principal components 
#	[<Error sampling rate>](optional): The error sampling rate [0-1] that is used for computing the error, It can be set to 0.01 to compute the error for only a small sample of the matrix, this speeds up the computations significantly 
# 	[<max iterations>] (optional): Maximum number of iterations before terminating, the default is 3
# 	[<output format>] (optional): One of three supported output formats (DENSE/COO/LIL), the default is DENSE. See Section Output Format for more details.
# 	[<0/1 (compute projected matrix or not)>] (optional):  0 or 1 value that specifies whether the user wants to project the input matrix on the principal components or not. 1 means that the projected matrix will be computed, and 0 means it will not be computed. The projected matrix is written in the output folder specified  by `-DOutput`

if [  $# -lt 1 ] 
	then 
		echo -e "\nUsage:\n$0 <master_url> \n" 
		exit 1
	fi 
 
master_url=$1  #master url has two options (local, spark://<IP>:7077) 
SCRIPT=$(readlink -f $0) # Absolute path to this script.
SCRIPTPATH=`dirname $SCRIPT` # Absolute path this script is in. /home/user/bin
$SPARK_HOME/bin/spark-submit --class org.qcri.sparkpca.SparkPCA --master $master_url --driver-java-options "-Di=$SCRIPTPATH/input/seqfiles -Do=$SCRIPTPATH/output -Drows=7 -Dcols=5 -Dpcs=3 -DerrSampleRate=1 -DmaxIter=3" $SCRIPTPATH/target/sparkPCA-1.0.jar 
