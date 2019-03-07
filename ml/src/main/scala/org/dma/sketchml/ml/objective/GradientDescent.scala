package org.dma.sketchml.ml.objective

import org.apache.flink.ml.math.DenseVector
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.data.DataSet
import org.dma.sketchml.ml.gradient._
import org.slf4j.{Logger, LoggerFactory}

object GradientDescent {
  private val logger: Logger = LoggerFactory.getLogger(GradientDescent.getClass)

  def apply(conf: MLConf): GradientDescent =
    new GradientDescent(conf.featureNum, conf.learnRate, conf.learnDecay, conf.batchSpRatio)


}

@SerialVersionUID(1113799434508676043L)
class GradientDescent(dim: Int, lr_0: Double, decay: Double, batchSpRatio: Double) extends Serializable {
  protected val logger: Logger = GradientDescent.logger

  var epoch: Int = 0
  var batch: Int = 0
  val batchNum: Double = Math.ceil(1.0 / batchSpRatio).toInt
  var count_updates_gradient = 0.0
  var accumulative_update_weight_gradient = 0.0
  var average_update_weight_gradient = 0.0

  def miniBatchGradientDescent(weight: DenseVector, dataSet: DataSet, loss: Loss): (Gradient, Int, Double, Double) = {
    if (dataSet == null) {
      return null
    }
    val startTime = System.currentTimeMillis()

    // WARNING: I changed logic here, so window size == batch size!
    val denseGrad = new DenseDoubleGradient(dim)
    var objLoss = 0.0
    val batchSize = dataSet.size
    for (i <- 0 until batchSize) {
      val ins = dataSet.loopingRead
      val pre = loss.predict(weight, ins.feature)
      val gradScala = loss.grad(pre, ins.label)
      denseGrad.plusBy(ins.feature, -1.0 * gradScala)
      objLoss += loss.loss(pre, ins.label)
    }
    val grad = denseGrad.toAuto
    grad.timesBy(1.0 / batchSize)
    objLoss=objLoss / batchSize

    if (loss.isL1Reg)
      l1Reg(grad, 0, loss.getRegParam)
    if (loss.isL2Reg)
      l2Reg(grad, weight, loss.getRegParam)
    val regLoss = loss.getReg(weight)

    //    logger.info(s"Epoch[$epoch] batch $batch gradient " +
    //      s"cost ${System.currentTimeMillis() - startTime} ms, "
    //      + s"batch size=$batchSize, obj loss=${objLoss / batchSize}, reg loss=$regLoss")
    batch += 1
    if (batch == batchNum) {
      epoch += 1
      batch = 0
    }
    (grad, batchSize, objLoss, regLoss)
  }

  private def l1Reg(grad: Gradient, alpha: Double, theta: Double): Unit = {
    val values = grad match {
      case dense: DenseDoubleGradient => dense.values
      case sparse: SparseDoubleGradient => sparse.values
      case _ => throw new UnsupportedOperationException(
        s"Cannot regularize ${grad.kind} kind of gradients")
    }
    if (values != null) {
      for (i <- values.indices) {
        if (values(i) >= 0 && values(i) <= theta)
          values(i) = (values(i) - alpha) max 0
        else if (values(i) < 0 && values(i) >= -theta)
          values(i) = (values(i) - alpha) min 0
      }
    }
  }

  private def l2Reg(grad: Gradient, weight: DenseVector, lambda: Double): Unit = {
    val w = weight.data
    grad match {
      case dense: DenseDoubleGradient => {
        val v = dense.values
        for (i <- v.indices)
          v(i) += w(i) * lambda
      }
      case sparse: SparseDoubleGradient => {
        val k = sparse.indices
        val v = sparse.values
        for (i <- k.indices)
          v(i) += w(k(i)) * lambda
      }
      case _ => throw new UnsupportedOperationException(
        s"Cannot regularize ${grad.kind} kind of gradients")
    }
  }

  def update(grad: Gradient, weight: DenseVector): Unit = {
    val startTime = System.currentTimeMillis()
    val lr = lr_0// / Math.sqrt(1.0 + decay * epoch)
    grad match {
      case dense: DenseDoubleGradient => update(dense, weight, lr)
      case sparse: SparseDoubleGradient => update(sparse, weight, lr)
      case dense: DenseFloatGradient => update(dense, weight, lr)
      case sparse: SparseFloatGradient => update(sparse, weight, lr)
      case sketchGrad: SketchGradient => update(sketchGrad.toAuto, weight)
      case fpGrad: FixedPointGradient => update(fpGrad.toAuto, weight)
      case zipGrad: ZipGradient => update(zipGrad.toAuto, weight)
    }


    //Calculate the average update weight instead of calculating weight for each single update
    count_updates_gradient += 1
    val temp_weight = System.currentTimeMillis() - startTime
    accumulative_update_weight_gradient += temp_weight
    average_update_weight_gradient = accumulative_update_weight_gradient / count_updates_gradient
    logger.info(s"Average update weights cost so far is $average_update_weight_gradient ms")

  }

  private def update(grad: DenseDoubleGradient, weight: DenseVector, lr: Double): Unit = {
    val g = grad.values
    val w = weight.data
    for (i <- w.indices)
      w(i) -= g(i) * lr
  }

  private def update(grad: SparseDoubleGradient, weight: DenseVector, lr: Double): Unit = {
    val k = grad.indices
    val v = grad.values
    val w = weight.data
    for (i <- k.indices)
      w(k(i)) -= v(i) * lr
  }

  private def update(grad: DenseFloatGradient, weight: DenseVector, lr: Double): Unit = {
    val g = grad.values
    val w = weight.data
    for (i <- w.indices)
      w(i) -= g(i) * lr
  }

  private def update(grad: SparseFloatGradient, weight: DenseVector, lr: Double): Unit = {
    val k = grad.indices
    val v = grad.values
    val w = weight.data
    for (i <- k.indices)
      w(k(i)) -= v(i) * lr
  }

}
