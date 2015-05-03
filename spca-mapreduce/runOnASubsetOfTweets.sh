#hadoop fs -rm -r output/* tmp/*
s=`date`

LOG=tweetlog$1.txt

hadoop jar target/spca-1.0-SNAPSHOT-job.jar org.qcri.pca.SPCADriver -D mapred.cluster.map.memory.mb=3072 -D mapred.job.map.memory.mb=3072 -D mapreduce.map.memory.mb=3072 -D mapred.task.timeout=6000000 -D mapred.healthChecker.script.timeout=30000 -D mapred.job.reuse.jvm.num.tasks=1  -i tweetseq/tweetsSeq$1 -o tweetoutput/$1 --tempDir tweettmp/$1 -rows 1024653032 -cols 71503 -pcs 50 -errRate 0.001 2>&1 | tee $LOG

e=`date`
echo $s
echo $e
echo $s >> $LOG
echo $e >> $LOG

. ./othercmd.sh
