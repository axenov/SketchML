package org.dma.sketchml.ml.algorithm

import hu.sztaki.ilab.ps.{FlinkParameterServer, WorkerLogic}
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.streaming.api.scala.function.AllWindowFunction
import org.apache.flink.streaming.api.scala.{DataStream, StreamExecutionEnvironment}
import org.apache.flink.streaming.api.windowing.windows.GlobalWindow
import org.apache.flink.util.Collector
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.data.{DataSet, LabeledData, Parser}
import org.dma.sketchml.ml.gradient.{DenseFloatGradient, Gradient}
import org.dma.sketchml.ml.objective.{GradientDescent, Loss}
import org.dma.sketchml.ml.parameterserver.GradientDistributionWorker
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer

object GeneralizedLinearModel {
  private val logger: Logger = LoggerFactory.getLogger(GeneralizedLinearModel.getClass)

  object Model {
    var weights: DenseVector = _
    var optimizer: GradientDescent = _
    var loss: Loss = _
    var gradient: Gradient = _
  }

  object Data {
    var trainData: DataSet = _
    var validData: DataSet = _
  }

}

import org.dma.sketchml.ml.algorithm.GeneralizedLinearModel.Data._
import org.dma.sketchml.ml.algorithm.GeneralizedLinearModel.Model._

abstract class GeneralizedLinearModel(protected val conf: MLConf, @transient protected val env: StreamExecutionEnvironment)
  extends Serializable {
  protected val logger: Logger = GeneralizedLinearModel.logger
  @transient protected var dataStream: DataStream[LabeledData] = _

  def loadData(): Unit = {
    val startTime = System.currentTimeMillis()
    dataStream = Parser.loadStreamData(conf.input, conf.format, conf.featureNum, conf.workerNum)(env)
    logger.info(s"Load data cost ${System.currentTimeMillis() - startTime} ms")
  }

  protected def initModel(): Unit

  def train(): Unit = {
    logger.info(s"Start to train a $getName model")
    logger.info(s"Configuration: $conf")
    val startTime = System.currentTimeMillis()
    initModel()

    val trainLosses = ArrayBuffer[Double](conf.epochNum)
    val validLosses = ArrayBuffer[Double](conf.epochNum)
    val timeElapsed = ArrayBuffer[Long](conf.epochNum)
    //    val batchNum = Math.ceil(1.0 / conf.batchSpRatio).toInt
    val baseLogic: DataStream[DataSet] = dataStream
      .countWindowAll(conf.windowSize)
      .apply(new ExtractTrainingData)(TypeInformation.of(classOf[DataSet]))

    val paramInit: Int => Gradient = (i: Int) => {
      System.out.println("GRADIENT INITIALIZED ON THE SERVER")
      new DenseFloatGradient(conf.featureNum)
    }
    val gradientUpdate: (Gradient, Gradient) => Gradient = (oldGradient: Gradient, newGradient: Gradient) => {
      System.out.println("GRADIENT UPDATED ON THE SERVER")
      Gradient.sum(conf.featureNum, Array(oldGradient, newGradient))
    }
    val workerLogic: WorkerLogic[DataSet, Int, Gradient, Gradient] = new GradientDistributionWorker(conf, optimizer, loss)


    // move this parameters to ParameterTool once it's confirmed everything works fine here
    val workerParallelism: Int = 1
    val psParallelism: Int = 1
    val iterationWaitTime: Long = 100


    FlinkParameterServer.transform[DataSet, Int, Gradient, Gradient](baseLogic, workerLogic, paramInit,
      gradientUpdate, workerParallelism, psParallelism, iterationWaitTime)(TypeInformation.of(classOf[DataSet]),
      TypeInformation.of(classOf[Int]),
      TypeInformation.of(classOf[Gradient]),
      TypeInformation.of(classOf[Gradient]))
  }

  def getName: String

}

class ExtractTrainingData extends AllWindowFunction[LabeledData, DataSet, GlobalWindow] {
  override def apply(window: GlobalWindow, input: Iterable[LabeledData], out: Collector[DataSet]): Unit = {
    trainData = new DataSet
    val it = input.iterator
    while (it.hasNext) {
      trainData += it.next()
    }
    out.collect(trainData)
  }
}