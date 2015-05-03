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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorView;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

import com.google.common.io.Closeables;

public final class MahoutCompatibilityTest extends PCATestCase {

  @Test
  public void testMAHOUT_1221() {
    // create a matrix with an unassigned row 0
    Matrix matrix = new SparseMatrix(1, 1);
    Vector view = matrix.viewRow(0);
    final double value = 1.23;
    view.assign(value);
    // test whether the update in the view is reflected in the matrix
    assertEquals("Matrix valye", view.getQuick(0), matrix.getQuick(0, 0),
        EPSILON);
  }

  @Test
  public void testMAHOUT_1238() throws IOException {
    Vector v = new SequentialAccessSparseVector(5);
    v.set(1, 3.0);
    v.set(3, 5.0);
    Vector view = new VectorView(v,0,v.size());
    doTestVectorWritableEquals(view);
  }
  
  private static void doTestVectorWritableEquals(Vector v) throws IOException {
    Writable vectorWritable = new VectorWritable(v);
    VectorWritable vectorWritable2 = new VectorWritable();
    writeAndRead(vectorWritable, vectorWritable2);
    Vector v2 = vectorWritable2.get();
    if (v instanceof NamedVector) {
      assertTrue(v2 instanceof NamedVector);
      NamedVector nv = (NamedVector) v;
      NamedVector nv2 = (NamedVector) v2;
      assertEquals(nv.getName(), nv2.getName());
      assertEquals("Victor", nv.getName());
    }
    assertEquals(v, v2);
  }

  private static void writeAndRead(Writable toWrite, Writable toRead) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutputStream dos = new DataOutputStream(baos);
    try {
      toWrite.write(dos);
    } finally {
      Closeables.close(dos, true);
    }

    ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
    DataInputStream dis = new DataInputStream(bais);
    try {
      toRead.readFields(dis);
    } finally {
      Closeables.close(dos, true);
    }
  }


}
