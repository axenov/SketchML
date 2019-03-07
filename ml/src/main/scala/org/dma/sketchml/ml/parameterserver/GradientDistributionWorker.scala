package org.dma.sketchml.ml.parameterserver

import hu.sztaki.ilab.ps.{ParameterServerClient, WorkerLogic}
import org.apache.flink.ml.math.DenseVector
import org.dma.sketchml.ml.common.Constants
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.data.DataSet
import org.dma.sketchml.ml.gradient.Gradient
import org.dma.sketchml.ml.objective.{GradientDescent, Loss}
import org.dma.sketchml.ml.util.ValidationUtil
import org.slf4j.{Logger, LoggerFactory}

class GradientDistributionWorker(conf: MLConf, optimizer: GradientDescent, loss: Loss) extends WorkerLogic[DataSet, Int, Gradient, Gradient] {
  private val logger: Logger = LoggerFactory.getLogger(GradientDistributionWorker.super.getClass)
  val startTimestamp: Long = System.currentTimeMillis()
  var weights: DenseVector = _
  var gradient: Gradient = _

  /**
    * Specifies logic for new incoming data.
    *
    * @param data
    * Incoming data.
    * @param ps
    * Interface to parameter server.
    */
  override def onRecv(data: DataSet, ps: ParameterServerClient[Int, Gradient, Gradient]): Unit = {
    // request pull from the server - it's asynchronous, we do not wait for the answer
    ps.pull(1)
    logger.info("ON NEW WINDOW")

    // weights initialization
    if (weights == null) {
      weights = new DenseVector(Array.fill(conf.featureNum) {
        scala.util.Random.nextDouble() * 0.001
      })
    }

    // validation on new window before it is used to training
    val validStart = System.currentTimeMillis()
    val (validLoss, truePos, trueNeg, falsePos, falseNeg, validNum, precision, trueRecall, falseRecall, aucResult) = ValidationUtil.calLossAucPrecision(weights, data, loss)
    val model: Unit = conf.algo match {
      case Constants.ML_LINEAR_REGRESSION => logger.info(s"Validation cost ${System.currentTimeMillis() - validStart} ms, " + s"valid size=$validNum, loss=$validLoss")
      case _ =>
        logger.info(s"Validation cost ${System.currentTimeMillis() - validStart} ms, "
          + s"loss=$validLoss, auc=$aucResult, precision=$precision, "
          + s"trueRecall=$trueRecall, falseRecall=$falseRecall")
        logger.info(s"PLOT::${System.currentTimeMillis() - startTimestamp},$validLoss,$aucResult")
    }

    // training
    val miniBathStart = System.currentTimeMillis()
    val (grad, _, _, _) =
      optimizer.miniBatchGradientDescent(weights, data, loss)

    if (gradient == null) {
      gradient = grad
    } else {
      gradient = Gradient.sum(conf.featureNum, Array(gradient, grad))
      gradient.timesBy(0.5)
    }
    optimizer.update(gradient, weights)
    logger.info(s"Calculation of local gradient and weights cost ${System.currentTimeMillis() - miniBathStart} ms")

    // push new value to the server
    ps.push(1, Gradient.compress(gradient, conf))
    logger.info("END OF WINDOW")
  }

  /**
    * Specifies the logic for incoming pull answers.
    *
    * @param paramId
    * ID of the received parameter.
    * @param paramValue
    * Value of the received parameter.
    * @param ps
    * Interface to parameter server.
    */
  override def onPullRecv(paramId: Int, paramValue: Gradient, ps: ParameterServerClient[Int, Gradient, Gradient]): Unit = {
    logger.info("ON PULL RECV")
    optimizer.update(paramValue, weights)
    logger.info("END ON PULL RECV")
  }
}