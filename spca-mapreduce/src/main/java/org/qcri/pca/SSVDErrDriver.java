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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

/**
 * @author maysam yabandeh
 */
public class SSVDErrDriver extends AbstractJob {

  public static final String VOPTION = "v";
  private static final String COLSOPTION = "D";
  private static final String PRINCIPALSOPTION = "d";
  private static final String ERRSAMPLE = "errRate";
  /**
   * The sampling rate that is used for computing the reconstruction error
   */
  private static float ERR_SAMPLE_RATE = 1.00f; // 100% default
  private int D;
  private int d;

  /**
   * @param args
   */
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new SSVDErrDriver(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(VOPTION, "v", "V matrix from SSVD run");
    addOption(COLSOPTION, "cols", "Number of cols");
    addOption(PRINCIPALSOPTION, "pcs", "Number of principal components");
    addOption(ERRSAMPLE, "errRate", "Sampling rate for computing the error (0-1]");
    if (parseArguments(args) == null) {
      return -1;
    }
    Path input = getInputPath();
    Path output = getOutputPath();
    Path vPath = new Path(getOption(VOPTION));
    D = Integer.parseInt(getOption(COLSOPTION));
    d = Integer.parseInt(getOption(PRINCIPALSOPTION));
    if (hasOption(ERRSAMPLE))
      ERR_SAMPLE_RATE = Float.parseFloat(getOption(ERRSAMPLE));

    Configuration conf = getConf();
    if (conf == null) {
      throw new IOException("No Hadoop configuration present");
    }
    run(conf, input, vPath, output);
    return 0;
  }

  private void run(Configuration conf, Path input, Path vPath, Path output) throws Exception {
    int round = 0;
    DistributedRowMatrix Ye = new DistributedRowMatrix(input, getTempPath(), 1, D);
    Ye.setConf(conf);
    DistributedRowMatrix V = new DistributedRowMatrix(vPath, getTempPath(), D, d);
    V.setConf(conf);
    DenseMatrix V_central = PCACommon.toDenseMatrix(V);
    
    DenseVector Ym = new DenseVector(Ye.numCols());
    MeanAndSpanJob masJob = new MeanAndSpanJob();
    masJob.compuateMeanAndSpan(Ye.getRowPath(), output, Ym,
        MeanAndSpanJob.NORMALIZE_MEAN, conf, ""+round);

    DenseVector Xm = new DenseVector(d);
    PCACommon.denseVectorTimesMatrix(Ym, V_central, Xm);
    DenseVector Zm = new DenseVector(D);
    PCACommon.vectorTimesMatrixTranspose(Xm, V_central, Zm);
    Zm = (DenseVector) Zm.minus(Ym);
    ReconstructionErrJob errJob = new ReconstructionErrJob();
    errJob.reconstructionErr(Ye, V, V, V_central, Ym, Xm, ERR_SAMPLE_RATE, conf, getTempPath(), ""+round);
  }
}



















