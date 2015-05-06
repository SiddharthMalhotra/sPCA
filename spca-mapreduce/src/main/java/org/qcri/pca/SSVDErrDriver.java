/**
 * QCRI, sPCA LICENSE
 * sPCA is a scalable implementation of Principal Component Analysis (PCA) on of Spark and MapReduce
 *
 * Copyright (c) 2015, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
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



















