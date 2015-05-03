package org.qcri.sparkpca;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class FileFormat {
	 private final static Logger log = LoggerFactory.getLogger(FileFormat.class);
	 public enum OutputFormat {
		DENSE,  //Dense matrix 
		LIL, //List of lists
		COO, //Coordinate List
	 } 
	 public enum InputFormat {
		DENSE,
		COO
	 } 
	 public static void main(String[] args) {
		final String inputPath;
		final int cardinality;
		final String outputPath;
		final InputFormat inputFormat;
		try {
			inputPath=System.getProperty("Input");
			if(inputPath==null)
				throw new IllegalArgumentException();
		}
		catch(Exception e) {
			printLogMessage("Input");
			return;
		}
		try {
			inputFormat=InputFormat.valueOf(System.getProperty("InputFmt"));
		}
		catch(IllegalArgumentException e) {
		    	 log.warn("Invalid Format " + System.getProperty("InputFmt") );
		    	 return;
		}
		catch(Exception e) {
		    	printLogMessage("InputFmt");
		    	return;
		}
		try {
			outputPath=System.getProperty("Output");
			if(outputPath==null)
				throw new IllegalArgumentException();
			File outputFile=new File(outputPath);
			if( outputFile.isFile() || outputFile==null )
			{
				log.error("Output Path must be a directory, " + outputPath + " is either not a directory or not a valid path");
				return;
			}
		}
		catch(Exception e) {
			printLogMessage("Output");
			return;
		}
		try {
			cardinality=Integer.parseInt(System.getProperty("Cardinality"));
		}
		catch(Exception e) {
			printLogMessage("Cardinality");
			return;
		}
		int base=-1;
		try {
			base=Integer.parseInt(System.getProperty("Base"));
		}
		catch(Exception e) {
			log.warn("It is not specified whether the input is zero-based or one-based, this parameter is useful only if the input is in COO format");
		}
		
		switch(inputFormat)
		{
			case COO:
				if(base==-1) {
					log.error("You have to specify whether the rows and columns IDs start with 0 or 1 using the argument -DBase");
					return;
				}
				convertFromCooToSeq(inputPath,cardinality,base,outputPath);
				break;
			case DENSE:
				convertFromDenseToSeq(inputPath,cardinality,outputPath);
				break;
		}
		
		
	}
	public static void convertFromDenseToSeq(String inputPath, int cardinality, String outputFolderPath)
	{
		try
    	{
	    	 final Configuration conf = new Configuration();
	         final FileSystem fs = FileSystem.get(conf);
	         SequenceFile.Writer writer;
	
	         final IntWritable key = new IntWritable();
	         final VectorWritable value = new VectorWritable();
	         
	         int lineNumber=0;
	         String thisLine;
	         File[] filePathList=null;
	         File inputFile=new File(inputPath);
	          if(inputFile.isFile()) // if it is a file
	          { 
	        	  filePathList= new File[1];
	        	  filePathList[0]=inputFile;
	          }
	          else
	          {
	        	  filePathList=inputFile.listFiles();
	          }
	          if(filePathList==null)
	          {
	        	  log.error("The path " + inputPath + " does not exist");
	          	  return;
	          }
	          for(File file:filePathList)
	          {
		          BufferedReader br = new BufferedReader(new FileReader(file));
		          Vector vector = null;
		          String outputFileName=outputFolderPath+ File.separator + file.getName() + ".seq";
		          writer=SequenceFile.createWriter(fs, conf, new Path(outputFileName), IntWritable.class, VectorWritable.class, CompressionType.BLOCK);
		          while ((thisLine = br.readLine()) != null) { // while loop begins here
		              if(thisLine.isEmpty())
		            	  continue;
		        	  String [] splitted = thisLine.split("\\s+");
		        	  vector = new SequentialAccessSparseVector(splitted.length);
		        	  for (int i=0; i < splitted.length; i++)
		        	  {
		        		  vector.set(i, Double.parseDouble(splitted[i]));
		        	  }
		        	  key.set(lineNumber);
		        	  value.set(vector);
		        	  //System.out.println(vector);
		        	  writer.append(key,value);//write last row
		        	  lineNumber++;
		          }
		          writer.close();
	          }   
		    }
	    	catch (Exception e) {
	    		e.printStackTrace();
	    	}
		
	    	
	}
	public static void convertFromCooToSeq(String inputPath, int cardinality, int base, String outputFolderPath){
    	try
    	{
    	 final Configuration conf = new Configuration();
         final FileSystem fs = FileSystem.get(conf);
         SequenceFile.Writer writer=null;

         final IntWritable key = new IntWritable();
         final VectorWritable value = new VectorWritable();
         
         Vector vector = null;
    
          String thisLine;
          
          int lineNumber=0;
          int prevRowID=-1;
          boolean first=true;
          File[] filePathList=null;
	      File inputFile=new File(inputPath);
          if(inputFile.isFile()) // if it is a file
          { 
        	  filePathList= new File[1];
        	  filePathList[0]=inputFile;
          }
          else
          {
        	  filePathList=inputFile.listFiles();
          }
          if(filePathList==null)
          {
        	  log.error("The path " + inputPath + " does not exist");
          	  return;
          }
          for(File file:filePathList)
          {
        	  BufferedReader br = new BufferedReader(new FileReader(file));
        	  String outputFileName=outputFolderPath+ File.separator + file.getName() + ".seq";
	          writer=SequenceFile.createWriter(fs, conf, new Path(outputFileName), IntWritable.class, VectorWritable.class, CompressionType.BLOCK);
	          while ((thisLine = br.readLine()) != null) { // while loop begins here   		   
	        	  String [] splitted = thisLine.split(",");
	        	  int rowID=Integer.parseInt(splitted[0]);
	        	  int colID=Integer.parseInt(splitted[1]);
	        	  double element=Double.parseDouble(splitted[2]);
	        	  if(first)
	        	  {
	        		  first=false;
	        		  vector = new SequentialAccessSparseVector(cardinality);
	        	  }
	        	  else if(rowID != prevRowID)
	        	  {
	        		  key.set(prevRowID);
	        		  value.set(vector);
	            	  //System.out.println(vector);
	            	  writer.append(key,value);//write last row
	            	  vector = new SequentialAccessSparseVector(cardinality);
	        	  }
	        	  prevRowID=rowID;
	        	  vector.set(colID-base,element);
	          }
	          /*//here we append the last vector in each file (assuming that we will start a new row in the next file
	          key.set(prevRowID);
	          value.set(vector);
	          //System.out.println("last vector");
	          //System.out.println(vector);
	    	  writer.append(key,value);//write last row
	          writer.close();
	          */
          }
          if(writer!=null) //append last vector in last file
          {
	          key.set(prevRowID);
	          value.set(vector);
	          //System.out.println("last vector");
	          //System.out.println(vector);
	    	  writer.append(key,value);//write last row
	          writer.close();
          }
          
    	}
    	catch (Exception e) {
    		e.printStackTrace();
    	}
    }
	private static void printLogMessage(String argName )
	 {
		log.error("Missing arguments -D" + argName);
		log.info("Usage: -DInput=<path/to/input/matrix> -DOutput=<path/to/outputfolder> -DInputFmt=<DENSE/COO> -DCardinaality=<number of columns> [-DBase=<0/1>(0 if input is zero-based, 1 if input is 1-based]"); 
	 }
	
}
