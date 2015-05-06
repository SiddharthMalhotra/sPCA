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

package org.qcri.pca;

import java.io.IOException;
import java.net.URL;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.qcri.pca.SPCADriver.InitialValues;
import org.junit.Before;
import org.junit.Test;
import org.junit.Assert;

/**
 * @author maysam yabandeh
 */
public class PCATest {// extends org.apache.mahout.common.MahoutTestCase {

  final static double EPSILON = 0.00001d;
  SPCADriver ppcaDriver;
  int N;//number of rows
  int D;//number of cols
  int d;//number of principal components
  Configuration conf;
  Path input;
  Path output;
  Path tmp;
  
  @Before
  public void setup() {
    ppcaDriver = new SPCADriver() {
      public Path getTempPath() {
        return tmp;
      }
    };
    N = 527;
    D = 38;
    d = 8;
    conf = new Configuration();
    conf.set("mapred.job.tracker", "local");
    conf.set("fs.default.name", "file:///");
    URL inputURL = this.getClass().getResource("/input.water");
    input = new Path(inputURL.toString());
    long currTime = System.currentTimeMillis();
    output = new Path("/tmp/" + currTime + "/output");
    tmp = new Path("/tmp/" + currTime + "/tmp");
    FileSystem fs;
    try {
      fs = FileSystem.get(output.toUri(), conf);
      fs.mkdirs(output);
      fs.mkdirs(tmp);
      fs.deleteOnExit(output);
      fs.deleteOnExit(tmp);
    } catch (IOException e) {
      e.printStackTrace();
      Assert.fail("Error in creating output direcoty " + output);
      return;
    }
  }
  
  @Test
  public void crossTestSequentialPPCAs() throws Exception {
    double jakobErr = ppcaDriver.runSequential_JacobVersion(conf, input,
        output, N, D, d);
    PCACommon.random = new Random(0);
    double bishopErr = ppcaDriver.runSequential(conf, input, 
        output, N, D, d);
    Assert.assertEquals(
        "The PPCA error between two sequntial methods is too different: "
            + jakobErr + "!= " + bishopErr, jakobErr, bishopErr, 0.01d);
  }
  
  @Test
  public void crossTestIterationOfMapReducePPCASequentialPPCA() throws Exception {
    Matrix C_central = PCACommon.randomMatrix(D, d);
    double ss = PCACommon.randSS();
    InitialValues initValSeq = new InitialValues(C_central, ss);
    InitialValues initValMR = new InitialValues(C_central.clone(), ss);

    //1. run sequential
    Matrix Ye_central = new DenseMatrix(N, D);
    int row = 0;
    for (VectorWritable vw : new SequenceFileDirValueIterable<VectorWritable>(
        input, PathType.LIST, null, conf)) {
      Ye_central.assignRow(row, vw.get());
      row++;
    }
    double bishopSeqErr = ppcaDriver.runSequential(conf, Ye_central, initValSeq, 1);
    
    //2. run mapreduce
    DistributedRowMatrix Ye = new DistributedRowMatrix(input, tmp, N, D);
    Ye.setConf(conf);
    double bishopMRErr = ppcaDriver.runMapReduce(conf, Ye, initValMR, output, N, D, d, 1, 1, 1, 1);
    
    Assert.assertEquals(
        "ss value is different in sequential and mapreduce PCA", initValSeq.ss,
        initValMR.ss, EPSILON);
    double seqCTrace = PCACommon.trace(initValSeq.C);
    double mrCTrace = PCACommon.trace(initValMR.C);
    Assert.assertEquals(
        "C value is different in sequential and mapreduce PCA", seqCTrace,
        mrCTrace, EPSILON);
    Assert.assertEquals(
        "The PPCA error between sequntial and mapreduce methods is too different: "
            + bishopSeqErr + "!= " + bishopMRErr, bishopSeqErr, bishopMRErr, EPSILON);
  }

  /* Too slow
  @Test
  public void crossTestMapReducePPCASequentialPPCA() throws Exception {
    double bishopSeqErr = ppcaDriver.runSequential(conf, input, 
        output, N, D, d);
    PCACommon.random = new Random(0);
    double bishopMRErr = ppcaDriver.runMapReduce(conf, input, 
        output, N, D, d, 1);
    Assert.assertEquals(
        "The PPCA error between sequntial and mapreduce methods is too different: "
            + bishopSeqErr + "!= " + bishopMRErr, bishopSeqErr, bishopMRErr, 0.01d);
  }
  */

}
