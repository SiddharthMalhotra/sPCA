file=$1
bunzip2 $file
file=${file%.bz2}
cat $file | perl NormalizeTweets.text.pl > $file.norm
rm $file


#file=${file}.norm
#hadoop fs -put $file tweets/
#rm $file
#bname=`basename $file`

#hadoop jar examples/target/mahout-examples-0.7-job.jar org.apache.mahout.pca.tweets.TweetParserJob -D mapreduce.map.memory.mb=3072 -D mapred.task.timeout=6000000 -D mapred.healthChecker.script.timeout=30000 -D mapred.job.reuse.jvm.num.tasks=1  -i tweets/$bname -o tweets/ 2>&1 | tee tweetlog${bname}.txt
