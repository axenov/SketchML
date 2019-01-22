package org.dma.sketchml.ml.algorithm

import org.dma.sketchml.ml.common.Constants
import org.dma.sketchml.ml.conf.MLConf
import org.slf4j.{Logger, LoggerFactory}

object LinearRegModel {
  private val logger: Logger = LoggerFactory.getLogger(LinearRegModel.getClass)

  def apply(conf: MLConf): LinearRegModel = new LinearRegModel(conf)

  def getName: String = Constants.ML_LINEAR_REGRESSION
}

class LinearRegModel(_conf: MLConf) extends GeneralizedLinearModel(_conf) {
  @transient override protected val logger: Logger = LinearRegModel.logger

  // TODO: How to do that?
  override protected def initModel(): Unit = {
//    executors.foreach(_ => {
//      weights = new DenseVector(new Array[Double](bcConf.value.featureNum))
//      optimizer = Adam(bcConf.value)
//      loss = new L2SquareLoss(bcConf.value.l2Reg)
//    })
  }

  override def getName: String = LinearRegModel.getName

}
