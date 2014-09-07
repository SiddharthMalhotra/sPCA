/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.qcri.sparkpca;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

/**
 * This code provides an implementation of PPCA: Probabilistic Principal
 * Component Analysis based on the paper from Tipping and Bishop:
 * 
 * sPCA: PPCA on top of Spark
 * 
 * 
 * @author Tarek Elgamal
 * 
 */

public class SparkPCA implements Serializable {

	 private final static Logger log = LoggerFactory.getLogger(SparkPCA.class);
	 final int MAX_ROUNDS=100;
     private final static boolean CALCULATE_ERR_ATTHEEND = false;

     public static void main(String[] args) {
    	 if(args.length < 5)
		 {
			 System.out.println("Usage: <path/to/input/matrix> <path/to/outputfile> <number of rows> <number of columns> <number of principal components> [Error sampling rate] [max iterations]");
			 return;
		 }
    	 //Parsing input arguments
		 String inputPath = args[0];
	     String outputPath = args[1];
	     final int nRows = Integer.parseInt(args[2]);
	     final int nCols = Integer.parseInt(args[3]);
	     final int nPCs   = Integer.parseInt(args[4]);
	     double errRate=1;
	     int maxIterations=3;
	     try {
	    	 errRate= Float.parseFloat(args[5]);
	     }
	     catch(Exception e) {
	    	
	    	 int length = String.valueOf(nRows).length();
	    	 if(length <= 4)
	    		 errRate= 1;
	         else
	        	 errRate=1/Math.pow(10, length-4);
	    	 log.warn("error sampling rate set to:  errRate=" + errRate);
	     }
	     try {
	    	 maxIterations=Integer.parseInt(args[6]);
	     }
	     catch(Exception e) {
	    	 log.warn("maximum iterations is set to default: maxIterations=" + maxIterations);
	     }
	     
	     //Setting Spark configuration parameters
	     SparkConf conf = new SparkConf().setAppName("SparkPCA");
	     conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
	     JavaSparkContext sc = new JavaSparkContext(conf);
	     
	     //compute principal components
	     org.apache.spark.mllib.linalg.Matrix principalComponentsMatrix = computePrincipalComponents(sc, inputPath, nRows, nCols, nPCs, errRate, maxIterations);
	     
	     //save principal components
	     PCAUtils.printMatrixToFile(principalComponentsMatrix, outputPath);
	     
	     log.info("Principal components computed successfully ");
	 }
     
     /**
      * Compute principal component analysis where the input is a RowMatrix
      * @param sc 
      * 	 		Spark context that contains the configuration parameters and represents connection to the cluster 
      * 			(used to create RDDs, accumulators and broadcast variables on that cluster)
      * @param inputMatrix
      * 			Input Matrix
      * @param nRows
      * 			Number of rows in input Matrix
      * @param nCols
      * 			Number of columns in input Matrix
      * @param nPCs
      * 			Number of desired principal components
      * @param errRate
      * 			The sampling rate that is used for computing the reconstruction error
      * @param maxIterations 
      * 			Maximum number of iterations before terminating
      * @return Matrix of size nCols X nPCs having the desired principal components
      */
     public static org.apache.spark.mllib.linalg.Matrix computePrincipalComponents(JavaSparkContext sc, RowMatrix inputMatrix, final int nRows, final int nCols, final int nPCs, final double errRate, final int maxIterations) {
    	  JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors = inputMatrix.rows().toJavaRDD().cache();
    	  return computePrincipalComponents(sc, vectors, nRows, nCols, nPCs, errRate, maxIterations);
     }
    
     /**
      * Compute principal component analysis where the input is a path for a hadoop sequence File <IntWritable key, VectorWritable value>
      * @param sc 
      * 	 		Spark context that contains the configuration parameters and represents connection to the cluster 
      * 			(used to create RDDs, accumulators and broadcast variables on that cluster)
      * @param inputPath
      * 			Path to the sequence file that represents the input matrix
      * @param nRows
      * 			Number of rows in input Matrix
      * @param nCols
      * 			Number of columns in input Matrix
      * @param nPCs
      * 			Number of desired principal components
      * @param errRate
      * 			The sampling rate that is used for computing the reconstruction error
      * @param maxIterations 
      * 			Maximum number of iterations before terminating
      * @return Matrix of size nCols X nPCs having the desired principal components
      */
     public static org.apache.spark.mllib.linalg.Matrix computePrincipalComponents(JavaSparkContext sc, String inputPath, final int nRows, final int nCols, final int nPCs, final double errRate, final int maxIterations) {
 	     
	    
	    //Read from sequence file
	    JavaPairRDD<IntWritable,VectorWritable> seqVectors = sc.sequenceFile(inputPath, IntWritable.class, VectorWritable.class);
	    
	    //Convert sequence file to RDD<org.apache.spark.mllib.linalg.Vector> of Vectors
	    JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors=seqVectors.map(new Function<Tuple2<IntWritable,VectorWritable>, org.apache.spark.mllib.linalg.Vector>() {

	                  public org.apache.spark.mllib.linalg.Vector call(Tuple2<IntWritable, VectorWritable> arg0)
	                                  throws Exception {

	                    org.apache.mahout.math.Vector mahoutVector = arg0._2.get();
	                    Iterator<Element> elements =  mahoutVector.nonZeroes().iterator();
	                    ArrayList<Tuple2<Integer, Double>> tupleList = new  ArrayList<Tuple2<Integer, Double>>();
	                    while(elements.hasNext())
	                    {
	                             Element e = elements.next();
	                             if(e.index() >= nCols || e.get() == 0 )
	                                continue;
	                             Tuple2<Integer,Double> tuple = new Tuple2<Integer,Double>(e.index(), e.get());
	                             tupleList.add(tuple);
	                    }
	                    org.apache.spark.mllib.linalg.Vector sparkVector = Vectors.sparse(nCols,tupleList);
	                    return sparkVector;
	              }
	    }).cache();
	    
	    return computePrincipalComponents(sc, vectors, nRows, nCols, nPCs, errRate, maxIterations);
     }
     /**
      * Compute principal component analysis where the input is an RDD<org.apache.spark.mllib.linalg.Vector> of vectors such that each vector represents a row in the matrix
      * @param sc 
      * 	 		Spark context that contains the configuration parameters and represents connection to the cluster 
      * 			(used to create RDDs, accumulators and broadcast variables on that cluster)
      * @param vectors
      * 			An RDD of vectors representing the rows of the input matrix
      * @param nRows
      * 			Number of rows in input Matrix
      * @param nCols
      * 			Number of columns in input Matrix
      * @param nPCs
      * 			Number of desired principal components
      * @param errRate
      * 			The sampling rate that is used for computing the reconstruction error
      * @param maxIterations 
      * 			Maximum number of iterations before terminating
      * @return Matrix of size nCols X nPCs having the desired principal components
      */
	 public static org.apache.spark.mllib.linalg.Matrix computePrincipalComponents(JavaSparkContext sc, JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors, final int nRows, final int nCols, final int nPCs, final double errRate, final int maxIterations) {
		 	     
	    System.out.println("Rows: " + nRows + ", Columns " + nCols);
	    double ss;
	    Matrix centralC;
	    
	   	// The two PPCA variables that improve over each iteration: Principal component matrix (C) and random variance (ss).
	    // Initialized randomly for the first iteration
	    ss= PCAUtils.randSS();
	    centralC= PCAUtils.randomMatrix(nCols, nPCs);
	    
		// 1. Mean Job : This job calculates the mean and span of the columns of the input RDD<org.apache.spark.mllib.linalg.Vector>
	    final Accumulator<double[]> matrixAccumY = sc.accumulator(new double[nCols], new VectorAccumulatorParam());
	    final double[] internalSumY=new double[nCols];
	    vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

	        public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0)
	                        throws Exception {
	        	org.apache.spark.mllib.linalg.Vector yi;
	        	int[] indices=null;
	        	int i;
	    		while(arg0.hasNext())
	    		{
	    			yi=arg0.next();
	    			indices=((SparseVector)yi).indices();
	    			for(i=0; i< indices.length; i++ )
    			    {
	    				internalSumY[indices[i]]+=yi.apply(indices[i]);
    			    }	                
	    		}
	    		matrixAccumY.add(internalSumY);
	        }

	  });//End Mean Job
	    
	  //Get the sum of column Vector from the accumulator and divide each element by the number of rows to get the mean
	  final Vector meanVector=new DenseVector(matrixAccumY.value()).divide(nRows);  
      final Broadcast<Vector> br_ym_mahout = sc.broadcast(meanVector);
      
      //2. Frobenious Norm Job : Obtain Frobenius norm of the input RDD<org.apache.spark.mllib.linalg.Vector>
      /**
       * To compute the norm2 of a sparse matrix, iterate over sparse items and sum
       * square of the difference. After processing each row, add the sum of the
       * meanSquare of the zero-elements that were ignored in the sparse iteration.
      */
      double meanSquareSumTmp=0;
      for(int i=0; i < br_ym_mahout.value().size(); i++)
      {
    	  double element=br_ym_mahout.value().getQuick(i);
    	  meanSquareSumTmp+=element*element;
      }
      final double meanSquareSum= meanSquareSumTmp;
      final Accumulator<Double> doubleAccumNorm2 = sc.accumulator(0.0);
      vectors.foreach( new VoidFunction<org.apache.spark.mllib.linalg.Vector>() {

         public void call(org.apache.spark.mllib.linalg.Vector yi) 
        		 throws Exception {
	          double norm2=0;
	          double meanSquareSumOfZeroElements = meanSquareSum;
	          int[] indices =  ((SparseVector)yi).indices();
	          int i;
	          int index;
	          double v;
	          //iterate over non
	          for(i=0; i < indices.length ; i++) {
	             index = indices[i];
	             v = yi.apply(index);
	             double mean = br_ym_mahout.value().getQuick(index);
	             double diff = v - mean;
	             diff *= diff;
	             
	             // cancel the effect of the non-zero element in meanSquareSum
	             meanSquareSumOfZeroElements -= mean * mean;
	             norm2 += diff;
	          }
	          // For all all zero items, the following has the sum of mean square
	          norm2+=meanSquareSumOfZeroElements;
	          doubleAccumNorm2.add(norm2);
         }

      });//end Frobenious Norm Job
      
      double norm2=doubleAccumNorm2.value();
      log.info("NOOOOORM2=" + norm2 );
      
	  // initial CtC  
	  Matrix centralCtC= centralC.transpose().times(centralC);
	  // -------------------------- EM Iterations
	  // while count
	  int round = 0;
	  double prevObjective = Double.MAX_VALUE;
	  double error = 0;
	  double relChangeInObjective = Double.MAX_VALUE;
	  final float threshold = 0.00001f;
	  final int firstRound=round;
	  for (; round <= maxIterations
	        && ((round - firstRound) <= 10 || relChangeInObjective > threshold); round++) {

		    // Sx = inv( ss * eye(d) + CtC );
		    Matrix centralSx = centralCtC.clone();
		    
		    // Sx = inv( eye(d) + CtC/ss );
		    centralSx.viewDiagonal().assign(Functions.plus(ss));
		    centralSx = PCAUtils.inv(centralSx);
		    	
		    // X = Y * C * Sx' => Y2X = C * Sx'
		    Matrix centralY2X =centralC.times(centralSx.transpose());
		    
		    //Broadcast Y2X because it will be used in several jobs and several iterations.
		    final Broadcast<Matrix> br_centralY2X = sc.broadcast(centralY2X);

			// Xm = Ym * Y2X
		    Vector xm_mahout = new DenseVector(nPCs);
		    xm_mahout = PCAUtils.denseVectorTimesMatrix(br_ym_mahout.value(), centralY2X, xm_mahout);
		    
		    //Broadcast Xm because it will be used in several iterations.
		    final Broadcast<Vector> br_xm_mahout = sc.broadcast(xm_mahout);
		    // We skip computing X as we generate it on demand using Y and Y2X
		    
			// 3. X'X and Y'X Job:  The job computes the two matrices X'X and Y'X
		    /**
		     * Xc = Yc * MEM	(MEM is the in-memory broadcasted matrix Y2X)
		     * 
		     * XtX = Xc' * Xc
		     * 
		     * YtX = Yc' * Xc
		     * 
		     * It also considers that Y is sparse and receives the mean vectors Ym and Xm
		     * separately.
		     * 
		     * Yc = Y - Ym
		     * 
		     * Xc = X - Xm
		     * 
		     * Xc = (Y - Ym) * MEM = Y * MEM - Ym * MEM = X - Xm
		     * 
		     * XtX = (X - Xm)' * (X - Xm)
		     * 
		     * YtX = (Y - Ym)' * (X - Xm)
		     * 
		    */
		    final Accumulator<double[][]> matrixAccumXtx = sc.accumulator(new double[nPCs][nPCs], new MatrixAccumulatorParam());
		    final Accumulator<double[][]> matrixAccumYtx = sc.accumulator(new double[nCols][nPCs], new MatrixAccumulatorParam());
		    final Accumulator<double[]> matrixAccumX = sc.accumulator(new double[nPCs], new VectorAccumulatorParam());
		    
		    /*
		     * Initialize the output matrices and vectors once in order to avoid generating massive intermediate data in the workers
		     */
		    final double[][] resArrayYtX = new double[nCols][nPCs];
		    final double[][] resArrayXtX = new double[nPCs][nPCs];
		    final double[]   resArrayX = new double[nPCs];
		    
		    /*
		     * Used to sum the vectors in one partition.
		    */
		    final double[][] internalSumYtX = new double[nCols][nPCs];
		    final double[][]   internalSumXtX = new double[nPCs][nPCs];
		    final double[]   internalSumX = new double[nPCs];
		   
		    vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

		    	public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
		    		org.apache.spark.mllib.linalg.Vector yi;
		    		while(arg0.hasNext())
		    		{
		    			yi=arg0.next();
		    			
		    			/*
		    		     * Perform in-memory matrix multiplication xi = yi' * Y2X
		    		    */
		    			PCAUtils.sparseVectorTimesMatrix(yi,br_centralY2X.value(), resArrayX);
		    			
		    			//get only the sparse indices
		    	        int[] indices=((SparseVector)yi).indices();
		    	        
                        PCAUtils.outerProductWithIndices(yi, br_ym_mahout.value(), resArrayX, br_xm_mahout.value(), resArrayYtX,indices);
		    	        PCAUtils.outerProductArrayInput(resArrayX, br_xm_mahout.value(), resArrayX, br_xm_mahout.value(), resArrayXtX);
		    	        int i,j,rowIndexYtX;
		    	        
		    	        //add the sparse indices only
		    	        for(i=0; i< indices.length; i++ )
		    			{
		    	            rowIndexYtX=indices[i];	
		    	        	for(j=0; j< nPCs; j++ )
		    	        	{
		    	        		internalSumYtX[rowIndexYtX][j]+=resArrayYtX[rowIndexYtX][j];
		    	        		resArrayYtX[rowIndexYtX][j]=0; //reset it
		    	        	}
		    	        	
		    			}
		    	        for(i=0; i< nPCs; i++ )
		    			{
		    	        	internalSumX[i]+=resArrayX[i];
		    	        	for(j=0; j< nPCs; j++ )
		    	        	{
		    	        		internalSumXtX[i][j]+=resArrayXtX[i][j];
		    	        		resArrayXtX[i][j]=0; //reset it
		    	        	}
		    	        	
		    			}
		    		}
		    		matrixAccumX.add(internalSumX);
		            matrixAccumXtx.add(internalSumXtX);
		            matrixAccumYtx.add(internalSumYtX);
		    	}
		    	
		    });//end  X'X and Y'X Job

		       
		   /*
		    * Get the values of the accumulators.
		   */
		   Matrix centralYtX = new DenseMatrix(matrixAccumYtx.value());
		   Matrix centralXtX = new DenseMatrix(matrixAccumXtx.value());
		   Vector centralSumX= new DenseVector(matrixAccumX.value());
		           
		   /*
	        * Mi = (Yi-Ym)' x (Xi-Xm) = Yi' x (Xi-Xm) - Ym' x (Xi-Xm)
	        * 
	        * M = Sum(Mi) = Sum(Yi' x (Xi-Xm)) - Ym' x (Sum(Xi)-N*Xm)
	        * 
	        * The first part is done in the previous job and the second in the following method
	       */         
		   centralYtX= PCAUtils.updateXtXAndYtx(centralYtX, centralSumX, br_ym_mahout.value(), xm_mahout, nRows);
		   centralXtX= PCAUtils.updateXtXAndYtx(centralXtX, centralSumX, xm_mahout, xm_mahout, nRows);
		        
		   // XtX = X'*X + ss * Sx
		   final double finalss = ss;
		   centralXtX.assign(centralSx, new DoubleDoubleFunction() {
		  	 	@Override
		  	 	public double apply(double arg1, double arg2) {
		  			return arg1 + finalss * arg2;
		  	 	}
		   });
		   
		   // C = (Ye'*X) / SumXtX;
		   Matrix invXtX_central = PCAUtils.inv(centralXtX);
		   centralC = centralYtX.times(invXtX_central);
		   centralCtC = centralC.transpose().times(centralC);
		    		
		   // Compute new value for ss
		   // ss = ( sum(sum(Ye.^2)) + trace(XtX*CtC) - 2sum(XiCtYit))/(N*D);
		   
		   //4. Variance Job: Computes part of variance that requires a distributed job
		   /**
		    * xcty = Sum (xi * C' * yi')
		    * 
		    * We also regenerate xi on demand by the following formula:
		    * 
		    * xi = yi * y2x
		    * 
		    * To make it efficient for uncentralized sparse inputs, we receive the mean
		    * separately:
		    * 
		    * xi = (yi - ym) * y2x = yi * y2x - xm, where xm = ym*y2x
		    * 
		    * xi * C' * (yi-ym)' = xi * ((yi-ym)*C)' = xi * (yi*C - ym*C)'
		    * 
		    */
		   
		   double ss2 = PCAUtils.trace(centralXtX.times(centralCtC));
		   final double[] resArrayYmC = new double[centralC.numCols()];
		   PCAUtils.denseVectorTimesMatrix(meanVector, centralC, resArrayYmC);
		   final Broadcast<Matrix> br_centralC = sc.broadcast(centralC);
		   final double[] resArrayYiC = new double[centralC.numCols()]; 	      
		   final Accumulator<Double> doubleAccumXctyt = sc.accumulator(0.0);
		   vectors.foreach( new VoidFunction<org.apache.spark.mllib.linalg.Vector>() {
		           public void call(org.apache.spark.mllib.linalg.Vector yi) throws Exception {
		               PCAUtils.sparseVectorTimesMatrix(yi, br_centralY2X.value(), resArrayX);
		               PCAUtils.sparseVectorTimesMatrix(yi, br_centralC.value(), resArrayYiC );
		                  
		               PCAUtils.subtract(resArrayYiC, resArrayYmC);
		               double dotRes = PCAUtils.dot(resArrayX,resArrayYiC); 
		               doubleAccumXctyt.add(dotRes);
		            }
		    }); //end Variance Job
		   
		    double xctyt = doubleAccumXctyt.value(); 
		    ss = (norm2 + ss2 - 2 * xctyt) / (nRows * nCols);
		    log.info("SSSSSSSSSSSSSSSSSSSSSSSSSSSS " + ss + " (" + norm2 + " + "
		                + ss2 + " -2* " + xctyt);
		      	
		    double objective = ss;
		    relChangeInObjective = Math.abs(1 - objective / prevObjective);
		    prevObjective = objective;
		    log.info("Objective:  %.6f    relative change: %.6f \n", objective,
		            relChangeInObjective);
		    
		    if (!CALCULATE_ERR_ATTHEEND) {
		          log.info("Computing the error at round " + round + " ...");
		          System.out.println("Computing the error at round " + round + " ...");
		         
		          double[] resArrayZmTmp = new double[centralC.numRows()];
		          PCAUtils.vectorTimesMatrixTranspose(xm_mahout, centralC, resArrayZmTmp);
		          final double[] resArrayZm= PCAUtils.subtractVectorArray(resArrayZmTmp , meanVector);
		          
		          //5. Reconstruction Error Job: The job computes the reconstruction error of the input matrix after updating the principal 
		          //components matrix
		          /**
		           * Xc = Yc * Y2X
		           * 
		           * ReconY = Xc * C'
		           * 
		           * Err = ReconY - Yc
		           * 
		           * Norm2(Err) = abs(Err).zSum().max()
		           * 
		           * To take the sparse matrixes into account we receive the mean separately:
		           * 
		           * X = (Y - Ym) * Y2X = X - Xm, where X=Y*Y2X and Xm=Ym*Y2X
		           * 
		           * ReconY = (X - Xm) * C' = X*C' - Xm*C'
		           * 
		           * Err = X*C' - Xm*C' - (Y - Ym) = X*C' - Y - Zm, where where Zm=Xm*C'-Ym
		           * 
		           */
		          
		          final Accumulator<double[]> vectorAccumErr = sc.accumulator(new double[nCols], new VectorAccumulatorAbsParam());
		          final Accumulator<double[]> vectorAccumNormCentralized = sc.accumulator(new double[nCols], new VectorAccumulatorAbsParam());
		          final double[] xiCtArray = new double[centralC.numRows()];		          
		          final double[] normalizedArray = new double[nCols];
		          
		          vectors.foreach( new VoidFunction<org.apache.spark.mllib.linalg.Vector>() {
		                  public void call(org.apache.spark.mllib.linalg.Vector yi) throws Exception {
		                          if (PCAUtils.pass(errRate))
		                          	return;
		                      
		                         PCAUtils.sparseVectorTimesMatrix(yi, br_centralY2X.value(), resArrayX);
		                         PCAUtils.vectorTimesMatrixTranspose(resArrayX, br_centralC.value(), xiCtArray);
		                         
		                         PCAUtils.denseVectorSubtractSparseSubtractDense(xiCtArray,yi,resArrayZm);
		                         vectorAccumErr.add(xiCtArray);
		                         
		                         PCAUtils.denseVectorMinusVector(yi, br_ym_mahout.value(), normalizedArray);
		                         vectorAccumNormCentralized.add(normalizedArray);
		                  }
		            }); // end Reconstruction Error Job

		          /*
		           * The norm of the reconstruction error matrix
		           */
		          double reconstructionError= PCAUtils.getMax(vectorAccumErr.value());
		          log.info("************************ReconsructionError=" + reconstructionError );
		          
		          /*
		           * The norm of the input matrix after centralization
		          */
		          double centralizedYNorm= PCAUtils.getMax(vectorAccumNormCentralized.value());
		          log.info("************************CentralizedNOrm=" + centralizedYNorm );
		          
		          error = reconstructionError/centralizedYNorm;
		          log.info("... end of computing the error at round " + round + " And error=" + error);
		      }
	    }
	    return PCAUtils.convertMahoutToSparkMatrix(centralC);
	  
	 }	
}
