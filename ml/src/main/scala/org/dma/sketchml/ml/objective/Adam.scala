package org.dma.sketchml.ml.objective

import org.apache.flink.ml.math.DenseVector
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.gradient._
import org.dma.sketchml.ml.objective.Adam.logger
import org.dma.sketchml.ml.util.Maths
import org.slf4j.{Logger, LoggerFactory}

object Adam {
  private val logger: Logger = LoggerFactory.getLogger(Adam.getClass)

  def apply(conf: MLConf): GradientDescent =
    new Adam(conf.featureNum, conf.learnRate, conf.learnDecay)

}

class Adam(dim: Int, lr_0: Double, decay: Double)
  extends GradientDescent(dim, lr_0, decay) {
  override protected val logger: Logger = Adam.logger

  val beta1 = 0.9
  val beta2 = 0.999
  var beta1_t = 0.9
  var beta2_t = 0.999
  val m = new Array[Double](dim)
  val v = new Array[Double](dim)
  var count_updates = 0.0
  var accumulative_update_weight = 0.0
  var average_update_weight = 0.0

  override def update(grad: Gradient, weight: DenseVector): Unit = {
    val startTime = System.currentTimeMillis()


    beta1_t *= beta1
    beta2_t *= beta2

    update0(grad, weight)

    //logger.info(s"Update weight cost ${System.currentTimeMillis() - startTime} ms")
    //Calcualte the average update wieght instead of calculating weight for each single update
    count_updates += 1
    val temp_weight = System.currentTimeMillis() - startTime
    accumulative_update_weight += temp_weight
    average_update_weight = accumulative_update_weight / count_updates
    logger.info(s"Average update weights cost so far is ${average_update_weight} ms")
  }

  private def update0(grad: Gradient, weight: DenseVector): Unit = {
    grad match {
      case dense: DenseDoubleGradient => update(dense, weight, lr_0)
      case sparse: SparseDoubleGradient => update(sparse, weight, lr_0)
      case dense: DenseFloatGradient => update(dense, weight, lr_0)
      case sparse: SparseFloatGradient => update(sparse, weight, lr_0)
      case sketchGrad: SketchGradient => update0(sketchGrad.toAuto, weight)
      case fpGrad: FixedPointGradient => update0(fpGrad.toAuto, weight)
      case zipGrad: ZipGradient => update0(zipGrad.toAuto, weight)
      case _ => throw new ClassNotFoundException(grad.getClass.getName)
    }
  }

  private def update(grad: DenseDoubleGradient, weight: DenseVector, lr: Double): Unit = {
    val g = grad.values
    val w = weight.data
    for (i <- w.indices) {
      val m_t = beta1 * m(i) + (1 - beta1) * g(i)
      val v_t = beta2 * v(i) + (1 - beta2) * g(i) * g(i)
      val newGrad = (Math.sqrt(1 - beta2_t) * m_t) / ((1 - beta1_t) * (Math.sqrt(v_t) + Maths.EPS))
      w(i) -= newGrad * lr
      m(i) = m_t
      v(i) = v_t
    }
  }

  private def update(grad: SparseDoubleGradient, weight: DenseVector, lr: Double): Unit = {
    val k = grad.indices
    val g = grad.values
    val w = weight.data
    for (i <- k.indices) {
      val dim = k(i)
      val grad = g(i)
      val m_t = beta1 * m(dim) + (1 - beta1) * grad
      val v_t = beta2 * v(dim) + (1 - beta2) * grad * grad
      val newGrad = (Math.sqrt(1 - beta2_t) * m_t) / ((1 - beta1_t) * (Math.sqrt(v_t) + Maths.EPS))
      w(dim) -= newGrad * lr
      m(dim) = m_t
      v(dim) = v_t
    }
  }

  private def update(grad: DenseFloatGradient, weight: DenseVector, lr: Double): Unit = {
    val g = grad.values
    val w = weight.data
    for (i <- w.indices) {
      val m_t = beta1 * m(i) + (1 - beta1) * g(i)
      val v_t = beta2 * v(i) + (1 - beta2) * g(i) * g(i)
      val newGrad = (Math.sqrt(1 - beta2_t) * m_t) / ((1 - beta1_t) * (Math.sqrt(v_t) + Maths.EPS))
      w(i) -= newGrad * lr
      m(i) = m_t
      v(i) = v_t
    }
  }

  private def update(grad: SparseFloatGradient, weight: DenseVector, lr: Double): Unit = {
    val k = grad.indices
    val g = grad.values
    val w = weight.data
    for (i <- k.indices) {
      val dim = k(i)
      val grad = g(i)
      val m_t = beta1 * m(dim) + (1 - beta1) * grad
      val v_t = beta2 * v(dim) + (1 - beta2) * grad * grad
      val newGrad = (Math.sqrt(1 - beta2_t) * m_t) / ((1 - beta1_t) * (Math.sqrt(v_t) + Maths.EPS))
      w(dim) -= newGrad * lr
      m(dim) = m_t
      v(dim) = v_t
    }
  }


}
