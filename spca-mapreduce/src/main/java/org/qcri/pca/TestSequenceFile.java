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
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;

/**
 * @author maysam yabandeh
 */
public class TestSequenceFile {

  /**
   * Generate SequnceFile format from a text file
   * 
   * @param args
   * @throws IOException
   */
  public static void main(String[] args) throws IOException {
    if (args.length < 1) {
      System.err.println("input text file is missing");
      return;
    }
    printSequenceFile(args[0], Integer.parseInt(args.length > 1 ? args[1] : "0"));
  }

  private static void printSequenceFile(String inputStr, int printRow) throws IOException {
    Configuration conf = new Configuration();
    Path finalNumberFile = new Path(inputStr);
    SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(conf),
        finalNumberFile, conf);
    IntWritable key = new IntWritable();
    VectorWritable value = new VectorWritable();
    Vector printVector = null;
    while (reader.next(key, value)) {
      if (key.get() == printRow)
        printVector = value.get();
      int cnt = 0;
      Iterator<Element> iter = value.get().nonZeroes().iterator();
      for (; iter.hasNext(); iter.next())
        cnt++;
      System.out.println("# "+ key + " " + cnt + " " + value.get().zSum());
    }
    reader.close();
    if (printVector != null)
      System.out.println("##### "+ printRow + " " + printVector);
    else
      System.out.println("##### "+ key + " " + value.get());
  }
}



























