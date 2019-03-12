package org.dma.sketchml.ml.objective

import org.apache.flink.ml.math.DenseVector
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.data.DataSet
import org.dma.sketchml.ml.gradient._
import org.slf4j.{Logger, LoggerFactory}

object GradientDescent {
  private val logger: Logger = LoggerFactory.getLogger(GradientDescent.getClass)

  def apply(conf: MLConf): GradientDescent =
    new GradientDescent(conf.featureNum)


}

@SerialVersionUID(1113799434508676043L)
class GradientDescent(dim: Int) extends Serializable {
  protected val logger: Logger = GradientDescent.logger

  def miniBatchGradientDescent(weight: DenseVector, dataSet: DataSet, loss: Loss): (Gradient, Int, Double, Double) = {
    if (dataSet == null) {
      return null
    }
    val startTime = System.currentTimeMillis()

    val denseGrad = new DenseDoubleGradient(dim)
    var objLoss = 0.0
    // batchSize is window size
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

    if (loss.isL1Reg)
      l1Reg(grad, 0, loss.getRegParam)
    if (loss.isL2Reg)
      l2Reg(grad, weight, loss.getRegParam)
    val regLoss = loss.getReg(weight)

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
}
