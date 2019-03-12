package org.dma.sketchml.ml.parameterserver

import hu.sztaki.ilab.ps.{ParameterServerClient, WorkerLogic}
import org.apache.flink.ml.math.DenseVector
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.data.DataSet
import org.dma.sketchml.ml.gradient.Gradient
import org.dma.sketchml.ml.objective.{GradientDescent, Loss}
import org.dma.sketchml.ml.util.ValidationUtil
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable

class GradientDistributionWorker(conf: MLConf, optimizer: GradientDescent, loss: Loss) extends WorkerLogic[DataSet, Int, Gradient, Gradient] {
  private val logger: Logger = LoggerFactory.getLogger(GradientDistributionWorker.super.getClass)
  val startTimestamp: Long = System.currentTimeMillis()
  val unpredictedData = new mutable.Queue[DataSet]()

  /**
    * Specifies logic for new incoming data.
    *
    * @param data
    * Incoming data.
    * @param ps
    * Interface to parameter server.
    */
  override def onRecv(data: DataSet, ps: ParameterServerClient[Int, Gradient, Gradient]): Unit = {
    // adds window to local queue
    unpredictedData.enqueue(data)
    val startWindowTimestamp: Long = System.currentTimeMillis()
    // request pull from the server - it's asynchronous, we do not wait for the answer
    ps.pull(1)
    logger.info(s"RunTime Per window accumulation costs (in ms): ${System.currentTimeMillis() - startWindowTimestamp}")
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
    // request pull from the server - it's asynchronous, we do not wait for the answer
    val startPullRecv: Long = System.currentTimeMillis()
    logger.info("ON PULL RECV")

    if (unpredictedData.isEmpty) {
      return
    }

    val data = unpredictedData.dequeue()
    val weights = new DenseVector(paramValue.toDense.values)

    // calculate gradient
    val (grad, _, _, _) = optimizer.miniBatchGradientDescent(weights, data, loss)
    // compress and push to server
    ps.push(1, Gradient.compress(grad, conf))
    logger.info(s"RunTime Per window costs (in ms):${System.currentTimeMillis() - startPullRecv},")

    // Validation
    val validStart: Long = System.currentTimeMillis()
    val (validLoss, truePos, trueNeg, falsePos, falseNeg, validNum, accuracy, trueRecall, falseRecall, aucResult, precision) = ValidationUtil.calLossAucPrecision(weights, data, loss)
    logger.info(s"Validation cost ${System.currentTimeMillis() - validStart} ms, " + s"loss=$validLoss, accuracy=$accuracy, auc=$aucResult, precision=$precision, " + s"trueRecall=$trueRecall, falseRecall=$falseRecall")
    logger.info(s"PLOT::${System.currentTimeMillis() - startTimestamp},$validLoss,$aucResult,$trueRecall,$falseRecall,$accuracy,$precision")
    logger.info("END ON PULL RECV")
  }
}