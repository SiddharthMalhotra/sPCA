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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

public class TweetParserUtil {
  HashSet<String> dictionary = new HashSet<String>();
  HashMap<String, Integer> baseDictionary = null;
  int cnt = 0;

  /**
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    TweetParserUtil parser = new TweetParserUtil();
    parser.createBaseDictionary();
    File f = new File(args[0]);
    parser.parseTweetFile(f);
    System.out.print(parser.dictionary.size() + " vs. " + parser.cnt);
  }

  public void createBaseDictionary() throws Exception {
    baseDictionary = new HashMap<String, Integer>();
    InputStream stream = this.getClass().getResourceAsStream("/words-ar.txt");
    BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
    String line = null;
    int id = 0;
    while ((line = reader.readLine()) != null) {
      baseDictionary.put(line, id++);
    }
    reader.close();
  }

  private void parseTweetFile(File f) throws Exception {
    @SuppressWarnings("resource")
    BufferedReader reader = new BufferedReader(new FileReader(f));
    String line = null;
    String[] tokens;
    final int TWEET_POS = 1;
    while ((line = reader.readLine()) != null) {
      tokens = line.split("\t");
      if (tokens.length == 1)
        continue;
      if (tokens.length != (TWEET_POS + 1))
        throw new Exception("Error in parsing the line with " + tokens.length
            + " tokens: " + line);
      parseTweetContent(tokens[TWEET_POS]);
    }
    reader.close();
  }

  java.util.Vector<Integer> theIndexes = new java.util.Vector<Integer>();

  Vector parseTweetContent(String line) {
    theIndexes.clear();
    String[] words = line.split(" ");
    for (String word : words) {
      char first = word.charAt(0);
      if (skip(first))
        continue;
      char last = word.charAt(word.length() - 1);
      if (skip(last))
        continue;
      if (word.startsWith("http:"))
        continue;
      if (word.indexOf('_') >= 0)
        word = word.replaceAll("_", "");
      cnt++;
      int index = getIndex(word);
      theIndexes.add(index);
    }
    SequentialAccessSparseVector vector = new SequentialAccessSparseVector(
        baseDictionary.size());
    for (int index : theIndexes)
      if (index >= 0)
        vector.set(index, 1);
    return vector;
  }

  boolean skip(char thechar) {
    if (thechar == '#')
      return true;
    if (thechar == '@')
      return true;
    if (thechar >= '0' && thechar <= '9')
      return true;
    if (thechar >= 'a' && thechar <= 'z')
      return true;
    if (thechar >= 'A' && thechar <= 'Z')
      return true;
    if (Character.isDigit(thechar))
      return true;
    return false;
  }

  int missword = 0;

  private int getIndex(String word) {
    Integer index = 0;
    if ((index = baseDictionary.get(word)) != null)
      return index;
    else if (word.length() > 5
        && (index = baseDictionary.get(word.substring(1))) != null)
      return index;
    else if (word.length() > 5
        && (index = baseDictionary.get(word.substring(0, word.length() - 1))) != null)
      return index;
    else if (word.length() > 6
        && (index = baseDictionary.get(word.substring(2))) != null)
      return index;
    else if (word.length() > 6
        && (index = baseDictionary.get(word.substring(0, word.length() - 2))) != null)
      return index;
    else if (word.length() > 7
        && (index = baseDictionary.get(word.substring(3))) != null)
      return index;
    else if (word.length() > 7
        && (index = baseDictionary.get(word.substring(0, word.length() - 3))) != null)
      return index;
    else if (word.length() > 6
        && (index = baseDictionary.get(word.substring(1, word.length() - 1))) != null)
      return index;
    else if (word.length() > 7
        && (index = baseDictionary.get(word.substring(1, word.length() - 2))) != null)
      return index;
    else if (word.length() > 7
        && (index = baseDictionary.get(word.substring(2, word.length() - 1))) != null)
      return index;
    else if (word.length() > 8
        && (index = baseDictionary.get(word.substring(2, word.length() - 2))) != null)
      return index;
    else
      missword++;
    return -1;
  }

}
