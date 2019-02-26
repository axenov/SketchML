package org.dma.sketchml.ml.parameterserver

import java.util.concurrent.Semaphore

import hu.sztaki.ilab.ps.{ParameterServerClient, WorkerLogic}
import org.apache.flink.ml.math.DenseVector
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.data.DataSet
import org.dma.sketchml.ml.gradient.Gradient
import org.dma.sketchml.ml.objective.{GradientDescent, Loss}
import org.dma.sketchml.ml.util.ValidationUtil
import org.slf4j.{Logger, LoggerFactory}

class GradientDistributionWorker(conf: MLConf, optimizer: GradientDescent, loss: Loss) extends WorkerLogic[DataSet, Int, Gradient, Gradient] {
  private val logger: Logger = LoggerFactory.getLogger(GradientDistributionWorker.super.getClass)

  var weights: DenseVector = _
  var gradient: Gradient = _

  /**
    * I'm not sure if this is working properly, because on local machine it seems that concurrent pushes happen before
    * a single pull is received back. This means gradient is not updated, but maybe in a real distributed environment it
    * will work better.
    */
  override def onRecv(data: DataSet, ps: ParameterServerClient[Int, Gradient, Gradient]): Unit = {
    ps.pull(1)
    if (weights == null) {
      weights = new DenseVector(new Array[Double](conf.featureNum))
    }
    val validStart = System.currentTimeMillis()
    val (validLoss, truePos, trueNeg, falsePos, falseNeg, validNum) = ValidationUtil.calLossPrecision(weights, data, loss)
    val precision = 1.0 * (truePos + trueNeg) / validNum
    val trueRecall = 1.0 * truePos / (truePos + falseNeg)
    val falseRecall = 1.0 * trueNeg / (trueNeg + falsePos)
    val (grad, _, _, _) =
      optimizer.miniBatchGradientDescent(weights, data, loss)
    logger.info(s"Validation cost ${System.currentTimeMillis() - validStart} ms, "
      + s"valid size=$validNum, loss=$validLoss, precision=$precision, "
      + s"trueRecall=$trueRecall, falseRecall=$falseRecall")

    if (gradient == null) {
      gradient = grad
    } else {
      gradient = Gradient.sum(conf.featureNum, Array(gradient, grad))
      gradient.timesBy(0.5)
    }

    optimizer.update(gradient, weights)

    ps.push(1, Gradient.compress(gradient, conf))
  }


  override def onPullRecv(paramId: Int, paramValue: Gradient, ps: ParameterServerClient[Int, Gradient, Gradient]): Unit = {
    logger.info("ON PULL RECV")
    if (paramValue == Gradient.zero) {
      logger.info("ZERO GRADIENT RECEIVED")
      return
    }
    logger.info("Update weights {}", weights)
    optimizer.update(paramValue, weights)
    logger.info("New weights {}", weights)
  }
}