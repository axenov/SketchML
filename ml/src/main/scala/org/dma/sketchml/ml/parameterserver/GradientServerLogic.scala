package org.dma.sketchml.ml.parameterserver

import java.util.concurrent.atomic.AtomicInteger

import hu.sztaki.ilab.ps.ParameterServer
import hu.sztaki.ilab.ps.server.SimplePSLogic
import org.dma.sketchml.ml.conf.MLConf
import org.dma.sketchml.ml.gradient.Gradient

class GradientServerLogic(paramInit: => Int => Gradient, paramUpdate: => (Gradient, Gradient) => Gradient, conf: MLConf)
  extends SimplePSLogic[Int, Gradient](paramInit, paramUpdate) {

  var gradientCounter: AtomicInteger = new AtomicInteger(0)

  override def onPullRecv(id: Int, workerPartitionIndex: Int, ps: ParameterServer[Int, Gradient, (Int, Gradient)]): Unit = {
    var value: Gradient = params.getOrElseUpdate(id, init(id))
    val currentGradientCounter = gradientCounter.get()
    if (currentGradientCounter > 1) {
      value.timesBy((currentGradientCounter - 1.0) / currentGradientCounter)
    }
    if (currentGradientCounter > 0) {
      value = Gradient.compress(value, conf)
    }

    ps.answerPull(id, value, workerPartitionIndex)
  }

  override def onPushRecv(id: Int, deltaUpdate: Gradient, ps: ParameterServer[Int, Gradient, (Int, Gradient)]): Unit = {
    gradientCounter.addAndGet(1)
    val c = params.get(id) match {
      case Some(q) =>
        update(q, deltaUpdate)
      case None =>
        deltaUpdate
    }
    params += ((id, c))
  }
}
