package org.dma.sketchml.ml.algorithm

import hu.sztaki.ilab.ps.{FlinkParameterServer, WorkerLogic}
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.streaming.api.scala.function.ProcessWindowFunction
import org.apache.flink.streaming.api.scala.{DataStream, StreamExecutionEnvironment}
import org.apache.flink.streaming.api.windowing.windows.GlobalWindow
import org.apache.flink.util.Collector
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.data.{DataSet, LabeledData, Parser}
import org.dma.sketchml.ml.gradient.{DenseFloatGradient, Gradient}
import org.dma.sketchml.ml.objective.{GradientDescent, Loss}
import org.dma.sketchml.ml.parameterserver.{GradientDistributionWorker, GradientServerLogic}
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

  var startTime: Long = _
  @transient protected var dataStream: DataStream[LabeledData] = _

  def loadData(): Unit = {
    //we don't need to check the loading time, as paper skip data loading time
    //val startTime = System.currentTimeMillis()
    dataStream = Parser.loadStreamData(conf.input, conf.format, conf.featureNum, conf.workerNum, negY = false)(env)
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
    startTime = System.currentTimeMillis()
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
      .process[DataSet](new ExtractTrainingData)(TypeInformation.of(classOf[DataSet]))

    /**
      * Logic used by each of the workers, defines behavior regarding receiving new data from incoming stream and data
      * received from the server as an answer to pull.
      */
    val workerLogic: WorkerLogic[DataSet, Int, Gradient, Gradient] = new GradientDistributionWorker(conf, optimizer, loss)

    // It's necessary to have only one copy of centralized gradient, thus we can only have 1 instance of the server.
    val psParallelism: Int = 1

    // It has to be 0, otherwise pulls are never received by the worker
    val iterationWaitTime: Long = 0

    /**
      * Parameter server API used to start the whole process.
      */
    FlinkParameterServer.transform(baseLogic, workerLogic, new GradientServerLogic(paramInit, gradientUpdate, conf),
      conf.workerNum, psParallelism, iterationWaitTime)(TypeInformation.of(classOf[DataSet]),
      TypeInformation.of(classOf[Int]),
      TypeInformation.of(classOf[Gradient]),
      TypeInformation.of(classOf[(Int, Gradient)]),
      TypeInformation.of(classOf[Gradient]))

    //end time for training process
    logger.info(s"Training run time is ${System.currentTimeMillis() - startTime} ms")
  }

  /**
    * First gradient initialization on the server - called on first pull request.
    */
  def paramInit: Int => Gradient = (i: Int) => {
    LoggerFactory.getLogger("Parameter server").info("GRADIENT INITIALIZED ON THE SERVER")
    new DenseFloatGradient(conf.featureNum)
  }

  /**
    * Logic used on the server when it receives gradient update.
    */
  def gradientUpdate: (Gradient, Gradient) => Gradient = (oldGradient: Gradient, update: Gradient) => {
    val logger = LoggerFactory.getLogger("Parameter server")
    logger.info("GRADIENT UPDATED ON THE SERVER")
    val updateStart = System.currentTimeMillis()
    val newGrad = Gradient.sum(conf.featureNum, Array(oldGradient, update))

    logger.info(s"Update and compression of gradient on the server cost ${System.currentTimeMillis() - updateStart} ms")
    logger.info(s"Training run time Up to update and compress gradient is ${System.currentTimeMillis() - startTime} ms")

    newGrad
  }

  def getName: String

}

@SerialVersionUID(1113799434508676099L)
class ExtractTrainingData extends ProcessWindowFunction[(Int, LabeledData), DataSet, Int, GlobalWindow] {

  /**
    * Groups incoming LabeledData into DataSet.
    */
  override def process(key: Int, context: Context, elements: Iterable[(Int, LabeledData)], out: Collector[DataSet]): Unit = {
    val trainData = new DataSet
    val it = elements.iterator
    while (it.hasNext) {
      val item = it.next()
      trainData.add(item._2)
    }
    out.collect(trainData)
  }
}