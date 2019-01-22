package org.dma.sketchml.ml.algorithm

import java.lang

import org.apache.flink.api.common.state.ReducingState
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.streaming.api.functions.ProcessFunction
import org.apache.flink.streaming.api.scala.function.{ProcessAllWindowFunction, ProcessWindowFunction}
import org.apache.flink.streaming.api.scala.{DataStream, StreamExecutionEnvironment}
import org.apache.flink.streaming.api.windowing.assigners.{GlobalWindows, TumblingProcessingTimeWindows}
import org.apache.flink.streaming.api.windowing.time.Time
import org.apache.flink.streaming.api.windowing.triggers.{CountTrigger, Trigger, TriggerResult}
import org.apache.flink.streaming.api.windowing.windows.{GlobalWindow, TimeWindow}
import org.apache.flink.util.Collector
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.data.{DataSet, LabeledData, Parser}
import org.dma.sketchml.ml.gradient.Gradient
import org.dma.sketchml.ml.objective.{GradientDescent, Loss}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

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

abstract class GeneralizedLinearModel(protected val conf: MLConf, @transient protected val env: StreamExecutionEnvironment) extends Serializable {
  @transient protected val logger: Logger = GeneralizedLinearModel.logger
  @transient protected var dataStream: DataStream[LabeledData] = _

  // TODO: How to distribute configuration between all workers?
  //  protected val bcConf: Broadcast[MLConf] = env.broadcast(conf)

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


    dataStream
      .countWindowAll(conf.windowSize)
      .process(new ComputePartialGradient)(TypeInformation.of(classOf[(Gradient, Int, Double, Double)]))
      .map((t: (Gradient, Int, Double, Double)) => (Gradient.compress(t._1, conf), t._2, t._3, t._4))(TypeInformation.of(classOf[(Gradient, Int, Double, Double)]))
      .process(new SendGradient)(TypeInformation.of(classOf[Unit]))
      .process(new GetNewGradient)(TypeInformation.of(classOf[Gradient]))
      .process(new UpdateWeights)(TypeInformation.of(classOf[Unit]))

    logger.info(s"Train done, total cost ${System.currentTimeMillis() - startTime} ms")
    logger.info(s"Train loss: [${trainLosses.mkString(", ")}]")
    logger.info(s"Valid loss: [${validLosses.mkString(", ")}]")
    logger.info(s"Time: [${timeElapsed.mkString(", ")}]")
  }

  //  protected def trainOneEpoch(epoch: Int, batchNum: Int): Double = {
  //    val epochStart = System.currentTimeMillis()
  //    var trainLoss = 0.0
  //    // One epoch means training of all batches
  //    for (batch <- 0 until batchNum) {
  //      val batchLoss = trainOneIteration(epoch, batch)
  //      trainLoss += batchLoss
  //    }
  //    val epochCost = System.currentTimeMillis() - epochStart
  //    logger.info(s"Epoch[$epoch] train cost $epochCost ms, loss=${trainLoss / batchNum}")
  //    // Result of one epoch training is an average of batches training
  //    trainLoss / batchNum
  //  }

  //  protected def trainOneIteration(epoch: Int, batch: Int): Double = {
  //    val batchStart = System.currentTimeMillis()
  //    // train one batch
  //    val batchLoss = computeGradient(epoch, batch)
  //    aggregateAndUpdate(epoch, batch)
  //    logger.info(s"Epoch[$epoch] batch $batch train cost "
  //      + s"${System.currentTimeMillis() - batchStart} ms")
  //    batchLoss
  //  }

  //  protected def computeGradient(epoch: Int, batch: Int): Double = {
  //    val miniBatchGDStart = System.currentTimeMillis()
  //    val (batchSize, objLoss, regLoss) = executors.aggregate(0, 0.0, 0.0)(
  //      seqOp = (_, _) => {
  //        // each of the executors computes gradient descent of its own piece of the data set
  //        val (grad, batchSize, objLoss, regLoss) =
  //          optimizer.miniBatchGradientDescent(weights, trainData, loss)
  //        // each executor has its own gradient value based on its partition of data
  //        gradient = grad
  //        (batchSize, objLoss, regLoss)
  //      },
  //      combOp = (c1, c2) => (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3)
  //    )
  //    // why regLess / conf.workerNum, where is this distributed among workers?
  //    val batchLoss = objLoss / batchSize + regLoss / conf.workerNum
  //    logger.info(s"Epoch[$epoch] batch $batch compute gradient cost "
  //      + s"${System.currentTimeMillis() - miniBatchGDStart} ms, "
  //      + s"batch size=$batchSize, batch loss=$batchLoss")
  //    batchLoss
  //  }

  // TODO: Method similar to aggregateAndUpdate, but related to the 'magic-box'
  // TODO: it should be invoked on new gradient appearing
  protected def consolidate(incomingGradient: Gradient): Unit = {
    val gradientsSum = Gradient.sum(conf.featureNum, Array(incomingGradient, gradient))
    // We should take the average value
    gradientsSum.timesBy(0.5)
    gradient = gradientsSum
    optimizer.update(gradient, weights)
  }

  //  protected def aggregateAndUpdate(epoch: Int, batch: Int): Unit = {
  //    val aggrStart = System.currentTimeMillis()
  //    // Based on number of features, sum partial compressed gradients to one gradient
  //    // result is dense or sparse gradient NOT SKETCH
  //    val sum = Gradient.sum(
  //      conf.featureNum,
  //      // Compression before sending gradients to driver
  //      executors.map(_ => Gradient.compress(gradient, bcConf.value)).collect()
  //    )
  //
  //    // Compressing summed up dense/sparse gradient to sketch before sending it to all workers
  //    val grad = Gradient.compress(sum, conf)
  //
  //    // Since we summed up partial gradients into one, we take average values.
  //    // no. of partial gradients = no. of workers =? no. of executors
  //    grad.timesBy(1.0 / conf.workerNum)
  //    logger.info(s"Epoch[$epoch] batch $batch aggregate gradients cost "
  //      + s"${System.currentTimeMillis() - aggrStart} ms")
  //
  //    val updateStart = System.currentTimeMillis()
  //    val bcGrad = sc.broadcast(grad)
  //    // Each executor updates weights based on the summed up and compressed gradient
  //    // Communication between driver and executor is via compressed gradient
  //    // Updating weights values is based on decompressed one
  //    executors.foreach(_ => optimizer.update(bcGrad.value, weights))
  //    logger.info(s"Epoch[$epoch] batch $batch update weights cost "
  //      + s"${System.currentTimeMillis() - updateStart} ms")
  //  }
  //
  //  // TODO: How to get validData? Assign some part of each window randomly? Will it work?
  //  protected def validate(epoch: Int): Double = {
  //    val validStart = System.currentTimeMillis()
  //    val (sumLoss, truePos, trueNeg, falsePos, falseNeg, validNum) =
  //      executors.aggregate((0.0, 0, 0, 0, 0, 0))(
  //        seqOp = (_, _) => ValidationUtil.calLossPrecision(weights, validData, loss),
  //        combOp = (c1, c2) => (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3,
  //          c1._4 + c2._4, c1._5 + c2._5, c1._6 + c2._6)
  //      )
  //    val validLoss = sumLoss / validNum
  //    val precision = 1.0 * (truePos + trueNeg) / validNum
  //    val trueRecall = 1.0 * truePos / (truePos + falseNeg)
  //    val falseRecall = 1.0 * trueNeg / (trueNeg + falsePos)
  //    logger.info(s"Epoch[$epoch] validation cost ${System.currentTimeMillis() - validStart} ms, "
  //      + s"valid size=$validNum, loss=$validLoss, precision=$precision, "
  //      + s"trueRecall=$trueRecall, falseRecall=$falseRecall")
  //    validLoss
  //  }

  def getName: String

}

class ComputePartialGradient extends ProcessAllWindowFunction[LabeledData, (Gradient, Int, Double, Double), GlobalWindow] {

  override def process(context: Context, elements: Iterable[LabeledData], out: Collector[(Gradient, Int, Double, Double)]): Unit = {
    trainData = new DataSet
    val it = elements.iterator
    while (it.hasNext) {
      trainData += it.next()
    }

    val (grad, batchSize, objLoss, regLoss) =
      optimizer.miniBatchGradientDescent(weights, trainData, loss)

    gradient = grad

    out.collect((grad, batchSize, objLoss, regLoss))
  }

}

class SendGradient extends ProcessFunction[(Gradient, Int, Double, Double), Unit] {
  override def processElement(value: (Gradient, Int, Double, Double), ctx: ProcessFunction[(Gradient, Int, Double, Double), Unit]#Context, out: Collector[Unit]): Unit = {
    // TODO: implement Atomix communication
  }
}

class GetNewGradient extends ProcessFunction[Unit, Gradient] {
  override def processElement(value: Unit, ctx: ProcessFunction[Unit, Gradient]#Context, out: Collector[Gradient]): Unit = {
    // Get new gradient
    // TODO: Implement Atomix communication
    // pass it to the next stream operator
    // out.collect()
    out.collect(gradient)
  }
}

class UpdateWeights extends ProcessFunction[Gradient, Unit] {
  override def processElement(value: Gradient, ctx: ProcessFunction[Gradient, Unit]#Context, out: Collector[Unit]): Unit = {
    optimizer.update(value, weights)
  }
}


