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

import java.util.List;
import junit.framework.Assert;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.qcri.pca.DummyRecordWriter;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

/**
 * @author maysam yabandeh
 */
public class MeanAndSpanJobTest extends PCATestCase {

  //try different combination of 0, NaN, +, -
  private final double[][] inputVectors = {
      { -0.49, Double.NaN, 0, -0.49, 3, -13.3 },
      { Double.NaN, -0.48, 0.45, 0.54, -20, 3.2 },
      { 0, 0.71, 0, 0.08, -12.2, 0.03 }, { 0.13, -0.06, -0.72, 0, 4, -0.00003 } };
  int cols = inputVectors[0].length;
  int rows = inputVectors.length;

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
    MeanAndSpanJob.MeanAndSpanMapper mapper = new MeanAndSpanJob.MeanAndSpanMapper();
    // construct the writers
    DummyRecordWriter<IntWritable, VectorWritable> mapWriter = new DummyRecordWriter<IntWritable, VectorWritable>();
    Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable>.Context context = DummyRecordWriter
        .build(mapper, conf, mapWriter);
    // perform the mapping
    for (int i = 0; i < inputVectors.length; i++) {
      VectorWritable row = new VectorWritable(array2Vector.get(inputVectors[i]));
      mapper.map(new IntWritable(i), row, context);
    }
    mapper.cleanup(context);
    verifyMapperOutput(mapWriter);

    MeanAndSpanJob.MeanAndSpanReducer reducer = new MeanAndSpanJob.MeanAndSpanReducer();
    // set up the writers
    DummyRecordWriter<IntWritable, VectorWritable> redWriter = new DummyRecordWriter<IntWritable, VectorWritable>();
    Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable>.Context redContext = DummyRecordWriter
        .build(reducer, conf, redWriter, IntWritable.class,
            VectorWritable.class);
    // perform the reduction
    for (IntWritable key : mapWriter.getKeys()) {
      reducer.reduce(key, mapWriter.getValue(key), redContext);
    }
    reducer.cleanup(redContext);
    verifyReducerOutput(redWriter);
  }

  private void verifyMapperOutput(
      DummyRecordWriter<IntWritable, VectorWritable> writer) {
    Assert.assertEquals("Each mapper should output three keys!", 3, writer
        .getKeys().size());
    for (IntWritable key : writer.getKeys()) {
      List<VectorWritable> list = writer.getValue(key);
      assertEquals("Mapper did not combine the results!", 1, list.size());
      Vector v = list.get(0).get();
      switch (key.get()) {
      case MeanAndSpanJob.MEANVECTOR:
        Assert.assertEquals("MeanVector size does not match!", v.size(),
            cols + 1);
        Assert.assertEquals("MeanVector count does not match!", rows, v.get(0),
            EPSILON);
        verifySum(inputVectors, v.viewPart(1, cols));
        break;
      case MeanAndSpanJob.MINVECTOR:
        Assert.assertEquals("MinVector size does not match!", v.size(), cols);
        verifyMin(inputVectors, v);
        break;
      case MeanAndSpanJob.MAXVECTOR:
        Assert.assertEquals("MaxVector size does not match!", v.size(), cols);
        verifyMax(inputVectors, v);
        break;
      default:
        Assert.fail("Unknown key from mapper");
      }
    }
  }

  private void verifyReducerOutput(
      DummyRecordWriter<IntWritable, VectorWritable> writer) {
    Assert.assertEquals("The reducer should output two keys!", 2, writer
        .getKeys().size());
    for (IntWritable key : writer.getKeys()) {
      List<VectorWritable> list = writer.getValue(key);
      assertEquals("Reducer did not combine the results!", 1, list.size());
      Vector v = list.get(0).get();
      switch (key.get()) {
      case MeanAndSpanJob.MEANVECTOR:
        Assert.assertEquals("MeanVector size does not match!", v.size(), cols);
        verifyMean(inputVectors, v);
        break;
      case MeanAndSpanJob.SPANVECTOR:
        Assert.assertEquals("SpanVector size does not match!", v.size(), cols);
        verifySpan(inputVectors, v);
        break;
      default:
        Assert.fail("Unknown key from mapper");
      }
    }
  }

  private void verifySpan(double[][] vectors, Vector spanVec) {
    for (int c = 0; c < cols; c++) {
      double max = NaN2Zero(vectors[0][c]);
      double min = max;
      for (int r = 0; r < rows; r++) {
        double val = vectors[r][c];
        max = Math.max(max, NaN2Zero(val));
        min = Math.min(min, NaN2Zero(val));
      }
      Assert.assertEquals("The span is incorrect: column: " + c, max - min,
          spanVec.get(c), EPSILON);
    }
  }

  private void verifyMax(double[][] vectors, Vector maxVec) {
    for (int c = 0; c < cols; c++) {
      double max = NaN2Zero(vectors[0][c]);
      for (int r = 0; r < rows; r++) {
        double val = vectors[r][c];
        max = Math.max(max, NaN2Zero(val));
      }
      Assert.assertEquals("The max is incorrect: column: " + c, max,
          maxVec.get(c), EPSILON);
    }
  }

  private void verifyMin(double[][] vectors, Vector minVec) {
    for (int c = 0; c < cols; c++) {
      double min = NaN2Zero(vectors[0][c]);
      for (int r = 0; r < rows; r++) {
        double val = vectors[r][c];
        min = Math.min(min, NaN2Zero(val));
      }
      Assert.assertEquals("The min is incorrect: column: " + c, min,
          minVec.get(c), EPSILON);
    }
  }

  private void verifySum(double[][] vectors, Vector sumVec) {
    for (int c = 0; c < cols; c++) {
      double sum = 0;
      for (int r = 0; r < rows; r++) {
        double val = vectors[r][c];
        sum += NaN2Zero(val);
      }
      Assert.assertEquals("The sum is incorrect: column: " + c, sum,
          sumVec.get(c), EPSILON);
    }
  }

  private void verifyMean(double[][] vectors, Vector meanVec) {
    for (int c = 0; c < cols; c++) {
      double sum = 0;
      for (int r = 0; r < rows; r++) {
        double val = vectors[r][c];
        sum += NaN2Zero(val);
      }
      double mean = sum / rows;
      Assert.assertEquals("The mean is incorrect: column: " + c, mean,
          meanVec.get(c), EPSILON);
    }
  }

  private double NaN2Zero(double val) {
    return Double.isNaN(val) ? 0 : val;
  }
}
