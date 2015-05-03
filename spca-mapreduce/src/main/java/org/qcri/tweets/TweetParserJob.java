/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.qcri.tweets;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobID;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.hadoop.mapreduce.TaskID;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

@SuppressWarnings("deprecation")
public class TweetParserJob extends AbstractJob {
  HashSet<String> dictionary = new HashSet<String>();
  HashMap<String, Integer> baseDictionary = null;
  int cnt = 0;

  /**
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new TweetParserJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }
    Configuration conf = getConf();
    if (conf == null) {
      throw new IOException("No Hadoop configuration present");
    }
    Path normFilePath = getInputPath();
    Path output = getOutputPath();

    run(conf, normFilePath, output);
    return 0;
  }

  public void run(Configuration initialConf, Path tweetsInputPath,
      Path matrixOutputPath) throws IOException, InterruptedException,
      ClassNotFoundException {
    JobConf conf = new JobConf(initialConf, TweetParserJob.class);
    conf.setJobName("TweetParser" + tweetsInputPath);
    Job job = new Job(conf);
    FileSystem fs = FileSystem.get(tweetsInputPath.toUri(), conf);
    tweetsInputPath = fs.makeQualified(tweetsInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, tweetsInputPath);
    job.setInputFormatClass(TextInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    System.out.println("OUTPUT --> " + matrixOutputPath.toString());
    job.setMapperClass(MyMapper.class);
    job.setNumReduceTasks(0);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setJarByClass(TweetParserJob.class);
    job.submit();
    job.waitForCompletion(true);
  }

  public static class MyMapper extends
      Mapper<LongWritable, Text, IntWritable, VectorWritable> {
    TweetParserUtil parser = new TweetParserUtil();

    @Override
    public void setup(Context context) throws IOException {
      try {
        parser.createBaseDictionary();
      } catch (Exception e) {
        e.printStackTrace();
      }
    }

    VectorWritable vw = new VectorWritable();
    IntWritable iw = new IntWritable();

    @Override
    public void map(LongWritable r, Text text, Context context)
        throws IOException, InterruptedException {
      String line = text.toString();
      String[] tokens = line.split("\t");
      if (tokens.length == 1)
        return;
      final int TWEET_POS = 1;
      if (tokens.length != (TWEET_POS + 1))
        throw new IOException("Error in parsing the line with " + tokens.length
            + " tokens: " + line);
      TaskAttemptID attemptid = context.getTaskAttemptID();
      int id = getId (attemptid);
      Vector columns = parser.parseTweetContent(tokens[TWEET_POS]);
      vw.set(columns);
      iw.set(id);
      context.write(iw, vw);
    }
    
    int lastSeq = 0;
    int getId(TaskAttemptID taskAttemptID) {
      TaskID taskid = taskAttemptID.getTaskID();
      JobID jobid = taskid.getJobID();
      int id = jobid.getId();
      id += id * 10 + taskid.getId();
      id += id * 4000000 + lastSeq++;
      return id;
    }
  }

}
