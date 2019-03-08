package org.dma.sketchml.ml.util

import org.apache.flink.ml.math.Vector


import org.dma.sketchml.ml.data.DataSet
import org.dma.sketchml.ml.objective.Loss
import org.dma.sketchml.sketch.util.Sort
import org.slf4j.{Logger, LoggerFactory}

@SerialVersionUID(1L)
object ValidationUtil extends Serializable {
  private val logger: Logger = LoggerFactory.getLogger(ValidationUtil.getClass)

  def calLossPrecision(weights: Vector, validData: DataSet, loss: Loss): (Double, Int, Int, Int, Int, Int, Double, Double, Double) = {
    val validStart = System.currentTimeMillis()
    val validNum = validData.size
    var validLoss = 0.0
    var truePos = 0 // ground truth: positive, prediction: positive
    var falsePos = 0 // ground truth: negative, prediction: positive
    var trueNeg = 0 // ground truth: negative, prediction: negative
    var falseNeg = 0 // ground truth: positive, prediction: negative

    for (i <- 0 until validNum) {
      val ins = validData.get(i)
      val pre = loss.predict(weights, ins.feature)
      if (pre * ins.label > 0) {
        if (pre > 0) truePos += 1
        else trueNeg += 1
      } else if (pre * ins.label <= 0) {
        if (pre > 0) falsePos += 1
        else falseNeg += 1
      }
      validLoss += loss.loss(pre, ins.label)
    }

    val precision = 1.0 * (truePos + trueNeg) / validNum
    val trueRecall = 1.0 * truePos / (truePos + falseNeg)
    val falseRecall = 1.0 * trueNeg / (trueNeg + falsePos)
    (validLoss, truePos, trueNeg, falsePos, falseNeg, validNum, precision, trueRecall, falseRecall)
  }

  def calLossAucPrecision(weights: Vector, validData: DataSet, loss: Loss): (Double, Int, Int, Int, Int, Int, Double, Double, Double, Double, Double) = {
    val validStart = System.currentTimeMillis()
    val validNum = validData.size
    var validLoss = 0.0
    val scoresArray = new Array[Double](validNum)
    val labelsArray = new Array[Double](validNum)
    var truePos = 0 // ground truth: positive, precision: positive
    var falsePos = 0 // ground truth: negative, precision: positive
    var trueNeg = 0 // ground truth: negative, precision: negative
    var falseNeg = 0 // ground truth: positive, precision: negative

    for (i <- 0 until validNum) {
      val ins = validData.get(i)
      val pre = loss.predict(weights, ins.feature)
      if (pre * ins.label > 0) {
        if (pre > 0) truePos += 1
        else trueNeg += 1
      } else if (pre * ins.label <= 0) {
        if (pre > 0) falsePos += 1
        else falseNeg += 1
      }
      scoresArray(i) = pre
      labelsArray(i) = ins.label
      validLoss += loss.loss(pre, ins.label)
    }

    validLoss = validLoss / validNum
    Sort.quickSort(scoresArray, labelsArray, 0, scoresArray.length)
    var M = 0L
    var N = 0L
    for (i <- 0 until validNum) {
      if (labelsArray(i) == 1)
        M += 1
      else
        N += 1
    }
    var sigma = 0.0
    for (i <- M + N - 1 to 0 by -1) {
      if (labelsArray(i.toInt) == 1.0)
        sigma += i
    }
    var aucResult = 0.0
    if (N != 0 && M != 0) {
      aucResult = (sigma - (M + 1) * M / 2) / M / N
    }

    val accuracy = 1.0 * (truePos + trueNeg) / validNum

    var trueRecall = 0.0
    if ((truePos + falseNeg) != 0) {
      trueRecall = 1.0 * truePos / (truePos + falseNeg)
    }

    var falseRecall = 0.0
    if ((trueNeg + falsePos) != 0) {
      falseRecall = 1.0 * trueNeg / (trueNeg + falsePos)
    }

    var precision = 0.0
    if ((truePos + falsePos) != 0) {
      precision = 1.0 * truePos / (truePos + falsePos)
    }

    //logger.info(truePos.toString)
    //logger.info(trueNeg.toString)
    //logger.info(falsePos.toString)
    //logger.info(falseNeg.toString)


    (validLoss, truePos, trueNeg, falsePos, falseNeg, validNum, accuracy, trueRecall, falseRecall, aucResult, precision)
  }
}