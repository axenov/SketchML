package org.dma.sketchml.ml.conf

import org.apache.flink.api.java.utils.ParameterTool
import org.dma.sketchml.ml.common.Constants._
import org.dma.sketchml.sketch.base.{Quantizer, SketchMLException}
import org.dma.sketchml.sketch.sketch.frequency.{GroupedMinMaxSketch, MinMaxSketch}

object MLConf {
  // ML Conf
  val ML_ALGORITHM: String = "spark.sketchml.algo"
  val ML_INPUT_PATH: String = "spark.sketchml.input.path"
  val ML_INPUT_FORMAT: String = "spark.sketchml.input.format"
  //val ML_TEST_DATA_PATH: String = "spark.sketchml.test.path"
  //val ML_NUM_CLASS: String = "spark.sketchml.class.num"
  //val DEFAULT_ML_NUM_CLASS: Int = 2
  val ML_NUM_WORKER: String = "spark.sketchml.worker.num"
  val ML_NUM_FEATURE: String = "spark.sketchml.feature.num"
  val ML_VALID_RATIO: String = "spark.sketchml.valid.ratio"
  val DEFAULT_ML_VALID_RATIO: Double = 0.25
  val ML_EPOCH_NUM: String = "spark.sketchml.epoch.num"
  val DEFAULT_ML_EPOCH_NUM: Int = 100
  val ML_BATCH_SAMPLE_RATIO: String = "spark.sketchml.batch.sample.ratio"
  val DEFAULT_ML_BATCH_SAMPLE_RATIO: Double = 0.1
  val ML_LEARN_RATE: String = "spark.sketchml.learn.rate"
  val DEFAULT_ML_LEARN_RATE: Double = 0.1
  val ML_LEARN_DECAY: String = "spark.sketchml.learn.decay"
  val DEFAULT_ML_LEARN_DECAY: Double = 0.9
  val ML_REG_L1: String = "spark.sketchml.reg.l1"
  val DEFAULT_ML_REG_L1: Double = 0.1
  val ML_REG_L2: String = "spark.sketchml.reg.l2"
  val DEFAULT_ML_REG_L2: Double = 0.1
  // Sketch Conf
  val SKETCH_GRADIENT_COMPRESSOR: String = "spark.sketchml.gradient.compressor"
  val DEFAULT_SKETCH_GRADIENT_COMPRESSOR: String = GRADIENT_COMPRESSOR_SKETCH
  val SKETCH_QUANTIZATION_BIN_NUM: String = "spark.sketchml.quantization.bin.num"
  val DEFAULT_SKETCH_QUANTIZATION_BIN_NUM: Int = Quantizer.DEFAULT_BIN_NUM
  val SKETCH_MINMAXSKETCH_GROUP_NUM: String = "spark.sketchml.minmaxsketch.group.num"
  val DEFAULT_SKETCH_MINMAXSKETCH_GROUP_NUM: Int = GroupedMinMaxSketch.DEFAULT_MINMAXSKETCH_GROUP_NUM
  val SKETCH_MINMAXSKETCH_ROW_NUM: String = "spark.sketchml.minmaxsketch.row.num"
  val DEFAULT_SKETCH_MINMAXSKETCH_ROW_NUM: Int = MinMaxSketch.DEFAULT_MINMAXSKETCH_ROW_NUM
  val SKETCH_MINMAXSKETCH_COL_RATIO: String = "spark.sketchml.minmaxsketch.col.ratio"
  val DEFAULT_SKETCH_MINMAXSKETCH_COL_RATIO: Double = GroupedMinMaxSketch.DEFAULT_MINMAXSKETCH_COL_RATIO
  // FixedPoint Conf
  val FIXED_POINT_BIT_NUM: String = "spark.sketchml.fixed.point.bit.num"
  val DEFAULT_FIXED_POINT_BIT_NUM = 8

  def apply(parameters: ParameterTool): MLConf = MLConf(
    parameters.get(ML_ALGORITHM),
    parameters.get(ML_INPUT_PATH),
    parameters.get(ML_INPUT_FORMAT),
    parameters.get(ML_NUM_WORKER).toInt,
    parameters.get(ML_NUM_FEATURE).toInt,
    parameters.getDouble(ML_VALID_RATIO, DEFAULT_ML_VALID_RATIO),
    parameters.getInt(ML_EPOCH_NUM, DEFAULT_ML_EPOCH_NUM),
    parameters.getDouble(ML_BATCH_SAMPLE_RATIO, DEFAULT_ML_BATCH_SAMPLE_RATIO),
    parameters.getDouble(ML_LEARN_RATE, DEFAULT_ML_LEARN_RATE),
    parameters.getDouble(ML_LEARN_DECAY, DEFAULT_ML_LEARN_DECAY),
    parameters.getDouble(ML_REG_L1, DEFAULT_ML_REG_L1),
    parameters.getDouble(ML_REG_L2, DEFAULT_ML_REG_L2),
    parameters.get(SKETCH_GRADIENT_COMPRESSOR, DEFAULT_SKETCH_GRADIENT_COMPRESSOR),
    parameters.getInt(SKETCH_QUANTIZATION_BIN_NUM, DEFAULT_SKETCH_QUANTIZATION_BIN_NUM),
    parameters.getInt(SKETCH_MINMAXSKETCH_GROUP_NUM, DEFAULT_SKETCH_MINMAXSKETCH_GROUP_NUM),
    parameters.getInt(SKETCH_MINMAXSKETCH_ROW_NUM, DEFAULT_SKETCH_MINMAXSKETCH_ROW_NUM),
    parameters.getDouble(SKETCH_MINMAXSKETCH_COL_RATIO, DEFAULT_SKETCH_MINMAXSKETCH_COL_RATIO),
    parameters.getInt(FIXED_POINT_BIT_NUM, DEFAULT_FIXED_POINT_BIT_NUM)
  )

}

case class MLConf(algo: String, input: String, format: String, workerNum: Int,
                  featureNum: Int, validRatio: Double, epochNum: Int,batchSpRatio: Double,
                  learnRate: Double, learnDecay: Double, l1Reg: Double, l2Reg: Double,
                  compressor: String, quantBinNum: Int, sketchGroupNum: Int,
                  sketchRowNum: Int, sketchColRatio: Double, fixedPointBitNum: Int) {
  require(Seq(ML_LOGISTIC_REGRESSION, ML_SUPPORT_VECTOR_MACHINE, ML_LINEAR_REGRESSION).contains(algo),
    throw new SketchMLException(s"Unsupported algorithm: $algo"))
  require(Seq(FORMAT_LIBSVM, FORMAT_CSV, FORMAT_DUMMY, FORMAT_LIBSVM_SEMICOLONS).contains(format),
    throw new SketchMLException(s"Unrecognizable file format: $format"))
  require(Seq(GRADIENT_COMPRESSOR_SKETCH, GRADIENT_COMPRESSOR_FIXED_POINT, GRADIENT_COMPRESSOR_ZIP,
    GRADIENT_COMPRESSOR_FLOAT, GRADIENT_COMPRESSOR_NONE).contains(compressor),
    throw new SketchMLException(s"Unrecognizable gradient compressor: $compressor"))

}

