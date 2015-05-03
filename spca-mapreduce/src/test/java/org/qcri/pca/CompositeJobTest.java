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
import java.util.List;

import junit.framework.Assert;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.qcri.pca.DummyRecordWriter;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.qcri.pca.CompositeJob.CompositeWritable;
import org.junit.Before;
import org.junit.Test;

/**
 * @author maysam yabandeh
 */
public class CompositeJobTest extends PCATestCase {

  // try different combination of 0, +, -
  // it assumes NaN are removed as part of normalization
  private final double[][] inputVectors = { { -0.49, 0, 3, -13.3 },
      { 0, -0.48, 0.45, 3.2 }, { 0.13, -0.06, 4, -0.00003 } };
  private final double[][] inMemMatrix = { { -0.49, 0 }, { 0, -0.48 },
      { 0, 0.71 }, { 4, -0.00003 } };
  final int xsize = inMemMatrix[0].length;
  private final double[][] xtx = { { 0, 0 }, { 0, 0 } };
  private final double[][] ytx = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } };
  int cols = inputVectors[0].length;
  int rows = inputVectors.length;
  double[] ym;
  private Configuration conf;
  private Path ymPath;
  private Path xmPath;
  private Path inMemMatrixPath;

  interface Array2Vector {
    Vector get(double[] array);
  }

  class Array2DenseVector implements Array2Vector {
    public Vector get(double[] array) {
      return new DenseVector(array);
    }
  }

  class Array2SparseVector implements Array2Vector {
    public Vector get(double[] array) {
      return new SequentialAccessSparseVector(new DenseVector(array));
    }
  }

  @Before
  public void setup() throws Exception {
    conf = new Configuration();
    long currTime = System.currentTimeMillis();
    Path outputDir = new Path("/tmp/" + currTime);
    FileSystem fs;
    try {
      fs = FileSystem.get(outputDir.toUri(), conf);
      fs.mkdirs(outputDir);
      fs.deleteOnExit(outputDir);
    } catch (IOException e) {
      e.printStackTrace();
      Assert.fail("Error in creating output direcoty " + outputDir);
      return;
    }
    ym = computeMean(inputVectors);
    double[] xm = new double[xsize];
    times(ym, inMemMatrix, xm);
    ymPath = PCACommon.toDistributedVector(new DenseVector(ym), outputDir,
        "ym", conf);
    xmPath = PCACommon.toDistributedVector(new DenseVector(xm), outputDir,
        "xm", conf);
    DistributedRowMatrix distMatrix = PCACommon.toDistributedRowMatrix(
        new DenseMatrix(inMemMatrix), outputDir, outputDir, "inMemMatrix");
    inMemMatrixPath = distMatrix.getRowPath();
    for (double[] row : xtx)
      for (int c = 0; c < row.length; c++)
        row[c] = 0;
    for (double[] row : ytx)
      for (int c = 0; c < row.length; c++)
        row[c] = 0;
    computeXtXandYtX(inputVectors);
  }

  @Test
  public void testWithDenseVectors() throws Exception {
    testJob(new Array2DenseVector());
  }

  @Test
  public void testWithSparseVectors() throws Exception {
    testJob(new Array2SparseVector());
  }

  void testJob(Array2Vector array2Vector) throws Exception {
    Configuration conf = new Configuration();
    conf.set(CompositeJob.MATRIXINMEMORY, inMemMatrixPath.toString());
    conf.setInt(CompositeJob.MATRIXINMEMORYROWS, inMemMatrix.length);
    conf.setInt(CompositeJob.MATRIXINMEMORYCOLS, xsize);
    Path xtxOutputPath = CompositeJob.getXtXPathBasedOnYm(ymPath);
    conf.set(CompositeJob.XTXPATH, xtxOutputPath.toString());
    conf.set(CompositeJob.YMPATH, ymPath.toString());
    conf.set(CompositeJob.XMPATH, xmPath.toString());
    CompositeJob.MyMapper mapper = new CompositeJob.MyMapper();
    // construct the writers
    DummyRecordWriter<CompositeWritable, VectorWritable> mapWriter = new DummyRecordWriter<CompositeWritable, VectorWritable>();
    Mapper<IntWritable, VectorWritable, CompositeWritable, VectorWritable>.Context context = DummyRecordWriter
        .build(mapper, conf, mapWriter);
    // perform the mapping
    mapper.setup(context);
    for (int i = 0; i < inputVectors.length; i++) {
      VectorWritable row = new VectorWritable(array2Vector.get(inputVectors[i]));
      mapper.map(new IntWritable(i), row, context);
    }
    mapper.cleanup(context);

    // perform the reducing
    CompositeJob.MyReducer reducer = new CompositeJob.MyReducer();
    DummyRecordWriter<IntWritable, VectorWritable> redWriter = new DummyRecordWriter<IntWritable, VectorWritable>();
    Reducer<CompositeWritable, VectorWritable, IntWritable, VectorWritable>.Context redContext = DummyRecordWriter
        .build(reducer, conf, redWriter, CompositeWritable.class,
            VectorWritable.class);
    reducer.setup(redContext);
    for (CompositeWritable key : mapWriter.getKeys()) {
      reducer.reduce(key, mapWriter.getValue(key), redContext);
    }

    verifyYtX(redWriter);
    CompositeJob compositeJob = new CompositeJob();
    compositeJob.loadXtX(ymPath, xsize, conf);
    verifyXtX(compositeJob.xtx);
  }

  private void verifyXtX(DenseMatrix xtxMatrix) {
    Assert.assertEquals("The computed xtx must have " + xsize + " rows!", xsize, xtxMatrix.numRows());
    Assert.assertEquals("The computed xtx must have " + xsize + " cols!", xsize, xtxMatrix.numCols());
    for (int r = 0; r < xsize; r++)
      for (int c = 0; c < xsize; c++)
        Assert.assertEquals("The xtx[" + r + "][" + c
            + "] is incorrect: ", xtx[r][c], xtxMatrix.get(r, c), EPSILON);
  }

  private void verifyYtX(
      DummyRecordWriter<IntWritable, VectorWritable> writer) {
    Assert.assertEquals("The reducer should output " + cols + " keys!", cols, writer
        .getKeys().size());
    for (IntWritable key : writer.getKeys()) {
      List<VectorWritable> list = writer.getValue(key);
      assertEquals("reducer produces more than one values per key!", 1,
          list.size());
      Vector v = list.get(0).get();
      assertEquals("reducer vector size must match the x size!", xsize,
          v.size());
      for (int c = 0; c < xsize; c++)
        Assert.assertEquals("The ytx[" + key.get() + "][" + c
            + "] is incorrect: ", ytx[key.get()][c], v.get(c), EPSILON);
    }
  }

  private void computeXtXandYtX(double[][] vectors) {
    double[][] normalizedVectors = vectors.clone();
    for (int i = 0; i < vectors.length; i++)
      normalizedVectors[i] = vectors[i].clone();
    vectors = normalizedVectors;
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
        vectors[r][c] -= ym[c];
    double[] xi = new double[xsize];
    for (int r = 0; r < rows; r++) {
      times(vectors[r], inMemMatrix, xi);
      for (int xRow = 0; xRow < xsize; xRow++)
        for (int xCol = 0; xCol < xsize; xCol++)
          xtx[xRow][xCol] += xi[xRow] * xi[xCol];
      for (int yCol = 0; yCol < cols; yCol++)
        for (int xCol = 0; xCol < xsize; xCol++)
          ytx[yCol][xCol] += vectors[r][yCol] * xi[xCol];
    }
  }

  private double[] computeMean(double[][] vectors) {
    double[] meanVector = new double[cols];
    for (int c = 0; c < cols; c++) {
      double sum = 0;
      for (int r = 0; r < rows; r++) {
        double val = vectors[r][c];
        sum += val;
      }
      meanVector[c] = sum / rows;
    }
    return meanVector;
  }

  private void times(double[] ym, double[][] matrix, double[] resVector) {
    int maxC = matrix[0].length;
    int maxR = matrix.length;
    for (int inMemC = 0; inMemC < maxC; inMemC++) {
      double sum = 0;
      for (int r = 0; r < maxR; r++) {
        double val = ym[r];
        double inMemVal = matrix[r][inMemC];
        sum += val * inMemVal;
      }
      resVector[inMemC] = sum;
    }
  }
}
