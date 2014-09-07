import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.*;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;




import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.linear.SparseFieldVector;
import org.apache.commons.math3.linear.SparseRealVector;
import org.apache.commons.math3.FieldElement;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math.linear.RealVector.Entry;
import org.apache.commons.math3.linear.SparseRealMatrix;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.DenseVector;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
//import org.apache.mahout.math.QRDecomposition;
import org.apache.mahout.math.Vector;
import org.jblas.DoubleMatrix;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;




//import org.apache.hadoop.io.IntWritable;
//import org.apache.hadoop.io.VectorWritable;

public class SparkPCADriver implements Serializable{

        private final static Logger log = LoggerFactory.getLogger(SparkPCADriver.class);
        final int MAX_ROUNDS=100;
        private final static boolean CALCULATE_ERR_ATTHEEND = false;


 

    public static void main(String[] args) {


    /*
    if (args.length < 8) {
      System.err.println("Usage: spca <local/cluster> <inputpath> <outputpath> <rows> <cols> <pcs> <sf> <errRate>");
      System.exit(1);
    }
    */

    /*
    JavaSparkContext sc = new JavaSparkContext(args[0], "sPCA",
      System.getenv("SPARK_HOME"), System.getenv("SPARK_EXAMPLES_JAR"));
    */


     String inputPath = args[0];
     String outputPath = args[1];
     int nRows = Integer.parseInt(args[2]);
     final int nCols = Integer.parseInt(args[3]);
     final int nPCs   = Integer.parseInt(args[4]);
     int sf   = Integer.parseInt(args[5]);
     final float errRate= Float.parseFloat(args[6]);
     int sampleRate=1; // No need to pass as args


    // JavaSparkContext sc = new JavaSparkContext("local", "Spark spca",
    //  "/home/qcri/spark-1.0.0", new String[]{"/home/qcri/sPCA/sparkPCA/target/spark-spca-1.0.jar"});
    //String path="mat.txt";
    SparkConf conf = new SparkConf().setAppName("Simple Application");
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
    JavaSparkContext sc = new JavaSparkContext(conf);

    //Read from sequence file
    JavaPairRDD<IntWritable,VectorWritable> seqVectors = sc.sequenceFile(inputPath, IntWritable.class, VectorWritable.class);
    JavaRDD<ExtendedRealVector> rows= seqVectors.map(new Function<Tuple2<IntWritable,VectorWritable>, ExtendedRealVector>() {

                public ExtendedRealVector call(Tuple2<IntWritable, VectorWritable> arg0)
                                throws Exception {

                        org.apache.mahout.math.Vector mahoutVector = arg0._2.get();
                        int vectorSize=mahoutVector.size();
                        double[] values = new double[nCols/*vectorSize*/];

                        for(int i=0; i < nCols/*mahoutVector.size()*/; i++)
                        {
                           values[i]= mahoutVector.get(i);
                        }
                        ExtendedRealVector realVector = new ExtendedRealVector(values); //MatrixUtils.createRealVector(values);
                        return realVector;

                }
        }).cache();


    // a JavaRDD of local vectors
    /*
    JavaRDD<Vector> vectors = sc.textFile(inputPath).map(
      new Function<String, Vector>() {
        public Vector call(String line) throws Exception {
          return parseVector(line);
        }
      }
    );
    */

    //Instead of matrix we can calculate the mean, min, max by having three agregator vectors (one, contains the sum and one contains max and one contains min)

    // Create a RowMatrix from an JavaRDD<Vector>.
   
    System.out.println("Rows: " + nRows + ", Columns " + nCols);

    //initialize Random C and random ss
    RealMatrix realCentralC= PCACommon.randomRealMatrix(nCols, nPCs);
    double ss= PCACommon.randSS();

    
    //Compute column statistics
    //Vector meanVector= distY.computeColumnSummaryStatistics().mean();
    final Accumulator<RealVector> matrixAccumY= sc.accumulator(MatrixUtils.createRealVector(new double[nCols]), new RealVectorAccumulatorParam());
    rows.foreach( new VoidFunction<ExtendedRealVector>() {
        public void call(ExtendedRealVector row) throws Exception {
        	matrixAccumY.add(MatrixUtils.createRealVector(row.toArray()));
       }
     });
    RealVector meanVector = matrixAccumY.value().mapDivide(nRows);
    
     //for()
    /*
    final Vector maxVector=  distY.computeColumnSummaryStatistics().max();
    final Vector minVector=  distY.computeColumnSummaryStatistics().min();
    */
   /*
    log.info("Mean Vector" +meanVector );
    log.info("Min Vector" +minVector );
    log.info("Max Vector" +maxVector );
    */
    //divide by the span
    /*
    JavaRDD<Vector> rows = vectors.map(
              new Function<Vector, Vector>() {

                                public Vector call(Vector arg0) throws Exception {
                                        int i =0;
                                        double[] originalV=arg0.toArray();
                                        double[] maxV= maxVector.toArray();
                                        double[] minV= minVector.toArray();

                                        double[] resultV=arg0.toArray();
                                        double span=0;
                                        for(i=0; i< arg0.toArray().length; i++)
                                        {
                                                span=(maxV[i] - minV[i]);
                                                if(span !=  0)
                                                        resultV[i] = originalV[i]/span;
                                                //resultV[i] =  originalV[i] / span != 0 ? span : 1;
                                        }
                                        return Vectors.dense(resultV);
                                }
              }
        ).cache();
        */

    //We can forget about dividing by span because it is always 1 in a binary matrix

    //JavaRDD<Vector> rows= vectors.cache();


    //get MeanVector
    //final RealVector ym = PCACommon.convertDenseSparkVectorToRealVector(meanVector);
    final Broadcast<RealVector> ym = sc.broadcast(meanVector);
    final Broadcast<DenseVector> ym_mahout = sc.broadcast(new DenseVector(meanVector.toArray()));

   

    //Compute Frobenious Norm
    double meanSquareSumTmp=0;
    for(int i=0; i < ym.value().getDimension(); i++)
    {
      double element=ym.value().getEntry(i);
      meanSquareSumTmp+=element*element;
    }
    final double meanSquareSum= meanSquareSumTmp;
    final Accumulator<Double> doubleAccumNorm2 = sc.accumulator(0.0);
    rows.foreach( new VoidFunction<ExtendedRealVector>() {
         public void call(ExtendedRealVector row) throws Exception {

                 double norm2=0;
                 double meanSquareSumOfZeroElements = meanSquareSum;
                 org.apache.commons.math.linear.RealVector yi = PCACommon.convertExtendedRealVectorToRealVectorMath(row);
                 Iterator<Entry> iterator =  yi.sparseIterator();
                   while (iterator.hasNext()) {
                    Entry element = iterator.next();
                    double v = element.getValue();
                    double mean = ym.value().getEntry(element.getIndex());
                    double diff = v - mean;
                    diff *= diff;
                    // cancel the effect of the non-zero element in meanSquareSum

                    meanSquareSumOfZeroElements -= mean * mean;
                    norm2 += diff;
                 }
                 norm2+=meanSquareSumOfZeroElements;
                 doubleAccumNorm2.add(norm2);


                 /*
                 double norm2=0;
                 RealVector yi = PCACommon.convertSparkVectorToRealVector(row);
                 RealVector yiminusym=yi.subtract(ym);
                 for(int i=0; i < yiminusym.getDimension(); i++)
                 {
                    norm2+= yiminusym.getEntry(i)*yiminusym.getEntry(i);
                 }
                 //norm2+=yiminusym.getNorm();
                 doubleAccumNorm2.add(norm2);
                 */
         }

    });//end Norm2Job
    double norm2=doubleAccumNorm2.value();
    log.info("NOOOOORM2=" + norm2 );
    RealMatrix realCentralCtC= realCentralC.transpose().multiply(realCentralC);
    int count = 1;
    double old = Double.MAX_VALUE;
    // -------------------------- EM Iterations
    // while count
    int round = 0;
    double prevObjective = Double.MAX_VALUE;
    double error = 0;
    double relChangeInObjective = Double.MAX_VALUE;
    final float threshold = 0.00001f;
    final int LAST_ROUND=1;
    final int firstRound=round;
    for (; round <= LAST_ROUND
            && ((round - firstRound) <= 10 || relChangeInObjective > threshold); round++) {

      // Sx = inv( ss * eye(d) + CtC );
      RealMatrix realCentralSx = realCentralCtC.copy();
      int ctcRows=realCentralCtC.getRowDimension();
      int ctcCols=realCentralCtC.getColumnDimension();
      if(ctcRows != ctcCols)
      {
          log.error("CtCentralC not a square matrix");
      }
      //log.info("CentralSX=" + realCentralSx);
      // Sx = inv( eye(d) + CtC/ss );
      realCentralSx= PCACommon.addToDiagonal(realCentralSx,ss);
      realCentralSx=new QRDecomposition(realCentralSx).getSolver().getInverse();
     //  log.info("CentralSX after inverse=" + realCentralSx);
     
      // X = Y * C * Sx' => Y2X = C * Sx'
      //final RealMatrix realCentralY2X=realCentralC.multiply(realCentralSx.transpose());
      final Broadcast<RealMatrix> realCentralY2X = sc.broadcast(realCentralC.multiply(realCentralSx.transpose()));
      
      ///////////////////////////Maath
      final org.apache.commons.math.linear.RealMatrix realCentralY2X_math= org.apache.commons.math.linear.MatrixUtils.createRealMatrix((realCentralY2X.value().getData()));
      final Broadcast<org.apache.commons.math.linear.RealMatrix> br_realCentralY2X_math = sc.broadcast(realCentralY2X_math);
      
      /////////////////////////Mahout
      final DenseMatrix realCentralY2X_mahout= new DenseMatrix(realCentralY2X.value().getData());
      final Broadcast<DenseMatrix> br_realCentralY2X_mahout = sc.broadcast(realCentralY2X_mahout);
      
      // log.info("CentralY2X (in memory matrix)=" + realCentralY2X);
      //Matrix centralY2X=PCACommon.convertRealMatrixToSparkMatrix(realCentralY2X);

      // Xm = Ym * Y2X
      ////////////Logging
      /*
      log.info("Ym dimensions=" + ym.getDimension());
      String text="";
      for(int i=0; i < ym.getDimension(); i++)
      {
          text+= ym.getEntry(i) + " ";

      }
      */
      //log.info("Ym data=" + text);
      final RealVector xm = realCentralY2X.value().preMultiply(ym.value());
      
      ////////////////Mahout
      DenseVector xm_mahout = new DenseVector(xm.toArray());
      final Broadcast<DenseVector> br_xm_mahout = sc.broadcast(xm_mahout);

      
      //log.info("Xm Vector=" + xm);

      final Accumulator<RealMatrix> matrixAccumXtx = sc.accumulator(MatrixUtils.createRealMatrix(nPCs,nPCs), new RealMatrixAccumulatorParam());
      final Accumulator<RealMatrix> matrixAccumYtx = sc.accumulator(MatrixUtils.createRealMatrix(nCols,nPCs), new RealMatrixAccumulatorParam());
      final Accumulator<RealVector> matrixAccumX = sc.accumulator(MatrixUtils.createRealVector(new double[nPCs]), new RealVectorAccumulatorParam());
     
      final Accumulator<Iterator<MatrixSlice>> matrixAccumXtx_mahout = sc.accumulator(new ArrayList<MatrixSlice>().iterator(), new MahoutMatrixAccumulatorParam());
      final Accumulator<Iterator<MatrixSlice>> matrixAccumYtx_mahout = sc.accumulator(new ArrayList<MatrixSlice>().iterator(), new MahoutMatrixAccumulatorParam());
      final Accumulator<Iterator<Vector.Element>> matrixAccumX_mahout = sc.accumulator(new ArrayList<Vector.Element>().iterator(), new MahoutVectorAccumulatorParam());
      

      //final DenseMatrix resMatrix =new DenseMatrix(nCols, nPCs); 

            rows.foreach(  new VoidFunction<ExtendedRealVector>() {
                public void call(ExtendedRealVector yi) throws Exception {
                                //ExtendedRealVector yi = PCACommon.convertSparkVectorToExtendedRealVector(row);
                             
                                //RealVector xi = realCentralY2X.preMultiply(yi);
                               //xi=xi.subtract(xm);
                	 		 
                       //////////////////Mathhhh
                       //org.apache.commons.math.linear.RealVector yi_math= new org.apache.commons.math.linear.OpenMapRealVector(yi.toArray());
                       //PCACommon.sparseVectorTimesMatrixMath(yi_math, br_realCentralY2X_math.value());
                      
                       //////////////////Mahout
                       DenseVector yi_mahout= new DenseVector(yi.toArray());
                       Vector xi_mahout= PCACommon.sparseVectorTimesMatrixManual(yi_mahout, br_realCentralY2X_mahout.value());
                       DenseMatrix resMatrix =new DenseMatrix(nCols, nPCs); 
                       Matrix centralYtxi =  PCACommon.outerProduct(yi_mahout, ym_mahout.value(), xi_mahout, br_xm_mahout.value(), resMatrix);
                	   Matrix centralXtxi = PCACommon.outerProduct(xi_mahout, br_xm_mahout.value(), xi_mahout, br_xm_mahout.value(), resMatrix);
                	   
                	    matrixAccumX_mahout.add(xi_mahout.all().iterator());		
                        matrixAccumXtx_mahout.add(centralXtxi.iterateAll());
                	 	matrixAccumYtx_mahout.add(centralYtxi.iterateAll());
                	 	
                                
                	 			//RealVector xi = PCACommon.sparseVectorTimesMatrix(yi, realCentralY2X.value());
                                /*
                                RealMatrix realCentralYtxi= PCACommon.outerProduct(yi,xi,xm);
                                RealMatrix realCentralXtxi= PCACommon.outerProduct(xi,xi,xm);
                                //RealMatrix realCentralYtxi= yi.outerProduct(xi).subtract(ym.outerProduct(xi));
                                //RealMatrix realCentralXtxi= xi.outerProduct(xi).subtract(xm.outerProduct(xi));
                                matrixAccumX.add(xi);
                                matrixAccumXtx.add(realCentralXtxi);
                                matrixAccumYtx.add(realCentralYtxi);
                                */
                        
                    }
             });
      // We skip computing X as we generate it on demand using Y and Y2X  (distX= distY.multiply(centralY2X);)

          log.info("Finisheeeeeeeeeed YtX, XtX at round " + round);
          System.out.println("Finisheeeeeeeeeed YtX, XtX at round " + round);
    
          RealVector realCentralSumX= matrixAccumX.value();
          RealMatrix realCentralXtx= matrixAccumXtx.value();
          RealMatrix realCentralYtx= matrixAccumYtx.value();
          
          /***
           * Mi = (Yi-Ym)' x (Xi-Xm) = Yi' x (Xi-Xm) - Ym' x (Xi-Xm)
           * 
           * M = Sum(Mi) = Sum(Yi' x (Xi-Xm)) - Ym' x (Sum(Xi)-N*Xm)
           * 
           * The first part is done in the previous rdd transformation and the second in the driver
           */
          //realCentralXtx = realCentralXtx.subtract(xm.outerProduct(realCentralSumX.subtract(xm.mapMultiply(nRows))));          
          realCentralYtx= PCACommon.updateXtXAndYtx(realCentralYtx, realCentralSumX, ym.value(), xm, nRows);
          realCentralXtx= PCACommon.updateXtXAndYtx(realCentralXtx, realCentralSumX, xm, xm, nRows);

          
          
          
          // XtX = X'*X + ss * Sx
          //log.info("CentraaaaaaaaaaaalXtx " + realCentralXtx);
          //log.info("CentraaaaaaaaaaaalYtx " + realCentralYtx);
          realCentralXtx=realCentralXtx.add(realCentralSx.scalarMultiply(ss));
          //log.info("CentraaaaaaaalXtx multiplied by Sx*ss " + realCentralXtx);
          //Sample Rate
          if (sampleRate < 1) { // rescale
                  realCentralXtx.scalarMultiply(1/sampleRate);
                  realCentralYtx.scalarMultiply(1/sampleRate);
          }

      // C = (Ye'*X) / XtX;
      RealMatrix invRealcentralXtX = new QRDecomposition(realCentralXtx).getSolver().getInverse();
      realCentralC = realCentralYtx.multiply(invRealcentralXtX);
      realCentralCtC = realCentralC.transpose().multiply(realCentralC);

       //log.info("CentraaaaaaaalXtx after inverse " + invRealcentralXtX);
       //log.info("CentraaaaaaaalC modified" + realCentralC);
       //log.info("CentraaaaaaaaaCTC " + realCentralCtC);
      // Compute new value for ss
      // ss = ( sum(sum(Ye.^2)) + PCACommon.trace(XtX*CtC) - 2sum(XiCtYit) )
      // /(N*D);
      final RealMatrix realCentralCTemp=realCentralC.copy();
      double ss2 = (realCentralXtx.multiply(realCentralCtC)).getTrace();
      final RealVector centralYmC= realCentralC.preMultiply(ym.value());
      final Accumulator<Double> doubleAccumXctyt = sc.accumulator(0.0);
      rows.foreach( new VoidFunction<ExtendedRealVector>() {
                public void call(ExtendedRealVector yi) throws Exception {
                //SparseRealVector yi = PCACommon.convertSparkVectorToRealVector(row);
                //RealVector xi = realCentralY2X.preMultiply(yi);
                //RealVector yiC=realCentralCTemp.preMultiply(yi);
                //ExtendedRealVector yi = PCACommon.convertSparkVectorToExtendedRealVector(row);
                RealVector xi = PCACommon.sparseVectorTimesMatrix(yi, realCentralY2X.value());
                RealVector yiC = PCACommon.sparseVectorTimesMatrix(yi, realCentralCTemp);
                
                yiC= yiC.subtract(centralYmC);
                double dotRes = xi.dotProduct(yiC);
                doubleAccumXctyt.add(dotRes);
                }
      }); //VarianceJob
      double xctyt = doubleAccumXctyt.value();
      if (sampleRate < 1) { // rescale
          xctyt = xctyt / sampleRate;
      }
      ss = (norm2 + ss2 - 2 * xctyt) / (nRows * nCols);

      //Logging
      log.info("SSSSSSSSSSSSSSSSSSSSSSSSSSSS " + ss + " (" + norm2 + " + "
              + ss2 + " -2* " + xctyt);
      System.out.println("SSSSSSSSSSSSSSSSSSSSSSSSSSSS " + ss + " (" + norm2 + " + "
              + ss2 + " -2* " + xctyt);
      /*
      double traceSx = realCentralSx.getTrace();
      double traceXtX = realCentralXtx.getTrace();
      double traceC = realCentralC.getTrace();
      double traceCtC = realCentralCtC.getTrace();
      log.info("TTTTTTTTTTTTTTTTT " + traceSx + " " + traceXtX + " "
              + traceC + " " + traceCtC);
      */

      double objective = ss;
      relChangeInObjective = Math.abs(1 - objective / prevObjective);
      prevObjective = objective;
      log.info("Objective:  %.6f    relative change: %.6f \n", objective,
          relChangeInObjective);
      if (!CALCULATE_ERR_ATTHEEND) {
        log.info("Computing the error at round " + round + " ...");
        System.out.println("Computing the error at round " + round + " ...");
        final Accumulator<RealVector> vectorAccumErr = sc.accumulator(MatrixUtils.createRealVector(new double[nCols]), new RealVectorAccumulatorAbsParam());
        final Accumulator<RealVector> vectorAccumNormCentralized = sc.accumulator(MatrixUtils.createRealVector(new double[nCols]), new RealVectorAccumulatorAbsParam());

        final RealVector zm= PCACommon.vectorTimesMatrixTranspose(xm,realCentralCTemp).subtract(ym.value());
        //log.info("Vectooooooooor Zm" + zm);

        rows.foreach( new VoidFunction<ExtendedRealVector>() {
                public void call(ExtendedRealVector yi) throws Exception {
                        if (PCACommon.pass(errRate))
                        	return;
                        //SparseRealVector yi = PCACommon.convertSparkVectorToRealVector(row);
                        //RealVector xi= realCentralY2X.preMultiply(yi);
                       //ExtendedRealVector yi = PCACommon.convertSparkVectorToExtendedRealVector(row);
                       RealVector xi = PCACommon.sparseVectorTimesMatrix(yi, realCentralY2X.value());
                       RealVector xiCt= PCACommon.vectorTimesMatrixTranspose(xi, realCentralCTemp);
                       RealVector errorVector= PCACommon.denseVectorSubtractSparseSubtractDense(xiCt,yi,zm);

                       vectorAccumErr.add(errorVector);
                       vectorAccumNormCentralized.add(PCACommon.sparseVectorMinusVector(yi,ym.value()));

                    //Add errorVector to sumOfError accumulator
                }
          }); //Reconstruction Job

        double reconstructionError= vectorAccumErr.value().getMaxValue();
        log.info("************************ReconsructionError=" + reconstructionError );
        double centralizedYNorm= vectorAccumNormCentralized.value().getMaxValue();
        log.info("************************CentralizedNOrm=" + centralizedYNorm );

        error = reconstructionError/centralizedYNorm; // ReconstructionJob //errJob.reconstructionErr(distY, distY2X, distC, centralC, ym, xm, ERR_SAMPLE_RATE, conf, getTempPath(), "" + round);
        log.info("... end of computing the error at round " + round + " And error=" + error);
      }

    }
    //return error;

  }

}
