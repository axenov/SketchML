package org.dma.sketchml.ml.algorithm

import hu.sztaki.ilab.ps.{FlinkParameterServer, WorkerLogic}
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.streaming.api.scala.function.{AllWindowFunction, ProcessWindowFunction, WindowFunction}
import org.apache.flink.streaming.api.scala.{DataStream, StreamExecutionEnvironment}
import org.apache.flink.streaming.api.windowing.windows.GlobalWindow
import org.apache.flink.util.Collector
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.data.{DataSet, LabeledData, Parser}
import org.dma.sketchml.ml.gradient.{DenseDoubleGradient, Gradient, Kind, SparseDoubleGradient}
import org.dma.sketchml.ml.objective.{GradientDescent, Loss}
import org.dma.sketchml.ml.parameterserver.GradientDistributionWorker
import org.slf4j.{Logger, LoggerFactory}

import scala.util.Random

object GeneralizedLinearModel {

  object Model {
    var optimizer: GradientDescent = _
    var loss: Loss = _
  }

}

import org.dma.sketchml.ml.algorithm.GeneralizedLinearModel.Model._

@SerialVersionUID(1113799434508676088L)
abstract class GeneralizedLinearModel(protected val conf: MLConf, @transient protected val env: StreamExecutionEnvironment)
  extends Serializable {
  protected val logger: Logger = LoggerFactory.getLogger(GeneralizedLinearModel.getClass)
  @transient protected var dataStream: DataStream[LabeledData] = _

  def loadData(): Unit = {
    //we don't need to check the loading time, as paper skip data loading time
    //val startTime = System.currentTimeMillis()
    dataStream = Parser.loadStreamData(conf.input, conf.format, conf.featureNum, conf.workerNum)(env)
    //logger.info(s"Load data cost ${System.currentTimeMillis() - startTime} ms")
  }

  protected def initModel(): Unit

  /**
    * Main method responsible for running training on streaming data.
    */
  def train(): Unit = {

    logger.info(s"Start to train a $getName model")
    logger.info(s"Configuration: $conf")
    //Start time for training process
    val startTime = System.currentTimeMillis()
    initModel()

    /**
      * Splitting incoming data into windows and extracting it to training data.
      */
    val baseLogic: DataStream[DataSet] = dataStream
      .map(item => {
        val key = Random.nextInt(conf.workerNum)
        (key, item)
      })(TypeInformation.of(classOf[(Int, LabeledData)]))
      .keyBy(t => t._1)(TypeInformation.of(classOf[Int]))
      .countWindow(conf.windowSize)
      .apply(new ExtractTrainingDataWindowFunction)(TypeInformation.of(classOf[DataSet]))


    /**
      * First gradient initialization on the server - called on first pull request.
      */
    val paramInit: Int => Gradient = (i: Int) => {
      LoggerFactory.getLogger("Parameter server").info("GRADIENT INITIALIZED ON THE SERVER")
      var weights= new DenseDoubleGradient(conf.featureNum)
      //************* Random initialization 
      //weights.plusBy(new DenseVector(Array.fill(conf.featureNum) { scala.util.Random.nextDouble() * 0.001}),1)
      weights
    }

    /**
      * Logic used on the server when it receives gradient update.
      */
    val gradientUpdate: (Gradient, Gradient) => Gradient = (weightsInGradient: Gradient, update: Gradient) => {
      val logger = LoggerFactory.getLogger("PARAMETER SERVER")
      val updateStartTime = System.currentTimeMillis()

      logger.info("WEIGHTS UPDATE")
      val weights = weightsInGradient.toDense.values
      // ************* Not Adam update
      val decompressedGradient = update.toAuto
      if (decompressedGradient.kind.equals(Kind.DenseDouble)) {
        val g = decompressedGradient.asInstanceOf[DenseDoubleGradient].values
        for (i <- weights.indices) {
          weights(i) -= g(i) * update.conf.learnRate
        }

      } else {
        val k = decompressedGradient.asInstanceOf[SparseDoubleGradient].indices
        val v = decompressedGradient.asInstanceOf[SparseDoubleGradient].values
        for (i <- k.indices)
          weights(k(i)) -= v(i) * update.conf.learnRate

      }
      //************* Adam update
      // optimizer.update(update,new DenseVector(weights))
      logger.info("END OF WEIGHTS UPDATE")
      logger.info(s"Weights update cost (in ms): ${System.currentTimeMillis() - updateStartTime}")

      new DenseDoubleGradient(weights.length, weights, update.conf)
    }

    /**
      * Logic used by each of the workers, defines behavior regarding receiving new data from incoming stream and data
      * received from the server as an answer to pull.
      */
    val workerLogic: WorkerLogic[DataSet, Int, Gradient, Gradient] = WorkerLogic.addPullLimiter(new GradientDistributionWorker(conf, optimizer, loss), 1)

    // It's necessary to have only one copy of centralized gradient, thus we can only have 1 instance of the server.
    val psParallelism: Int = 1

    // It has to be 0, otherwise pulls are never received by the worker
    val iterationWaitTime: Long = 0

    /**
      * Parameter server API used to start the whole process.
      */
    FlinkParameterServer.transform[DataSet, Int, Gradient, Gradient](baseLogic, workerLogic, paramInit,
      gradientUpdate, conf.workerNum, psParallelism, iterationWaitTime)(TypeInformation.of(classOf[DataSet]),
      TypeInformation.of(classOf[Int]),
      TypeInformation.of(classOf[Gradient]),
      TypeInformation.of(classOf[Gradient]))

    //end time for training process
    logger.info(s"Training run time is ${System.currentTimeMillis() - startTime} ms")
  }

  def getName: String

}

@SerialVersionUID(1113799434508676095L)
class ExtractTrainingDataWindowAll extends AllWindowFunction[LabeledData, DataSet, GlobalWindow] {

  /**
    * Groups incoming LabeledData into DataSet.
    */
  override def apply(window: GlobalWindow, input: Iterable[LabeledData], out: Collector[DataSet]): Unit = {
    val trainData = new DataSet
    val it = input.iterator
    while (it.hasNext) {
      val item = it.next()
      trainData.add(item)
    }
    out.collect(trainData)
  }
}

@SerialVersionUID(1113799434508676012L)
class ExtractTrainingDataWindowFunction extends WindowFunction[(Int, LabeledData), DataSet, Int, GlobalWindow] {

  /**
    * Groups incoming LabeledData into DataSet.
    */
  override def apply(key: Int, window: GlobalWindow, input: Iterable[(Int, LabeledData)], out: Collector[DataSet]): Unit = {
    val trainData = new DataSet
    val it = input.iterator
    while (it.hasNext) {
      val item = it.next()
      trainData.add(item._2)
    }
    out.collect(trainData)
  }
}
