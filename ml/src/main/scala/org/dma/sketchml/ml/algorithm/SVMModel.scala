package org.dma.sketchml.ml.algorithm

import org.apache.flink.streaming.api.scala.StreamExecutionEnvironment
import org.dma.sketchml.ml.algorithm.GeneralizedLinearModel.Model.{loss, optimizer}
import org.dma.sketchml.ml.common.Constants
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.objective.{Adam, GradientDescent, L2HingeLoss}
import org.slf4j.{Logger, LoggerFactory}

object SVMModel {
  private val logger: Logger = LoggerFactory.getLogger(SVMModel.getClass)

  def apply(conf: MLConf, env: StreamExecutionEnvironment): SVMModel = new SVMModel(conf, env)

  def getName: String = Constants.ML_SUPPORT_VECTOR_MACHINE
}

class SVMModel(_conf: MLConf, _env: StreamExecutionEnvironment) extends GeneralizedLinearModel(_conf, _env) {
  @transient override protected val logger: Logger = SVMModel.logger

  override protected def initModel(): Unit = {
    optimizer = Adam(_conf)
    loss = new L2HingeLoss(_conf.l2Reg)
  }

  override def getName: String = SVMModel.getName

}
