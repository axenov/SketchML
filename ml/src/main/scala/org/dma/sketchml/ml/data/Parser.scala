package org.dma.sketchml.ml.data

import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.ml.math.SparseVector
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.streaming.api.scala.StreamExecutionEnvironment
import org.apache.flink.streaming.api.scala.DataStream
import org.dma.sketchml.ml.common.Constants
import org.dma.sketchml.ml.util.Maths


@SerialVersionUID(1L)
object Parser extends Serializable {
  def loadStreamData(input: String, format: String, maxDim: Int, numPartition: Int,
                     negY: Boolean = true)(implicit sc: StreamExecutionEnvironment): DataStream[LabeledData] = {
    val parse: (String, Int, Boolean) => LabeledData = format match {
      case Constants.FORMAT_LIBSVM => Parser.parseLibSVM
      case Constants.FORMAT_CSV => Parser.parseCSV
      case Constants.FORMAT_DUMMY => Parser.parseDummy
      case Constants.FORMAT_LIBSVM_SEMICOLONS => Parser.parseLibSVMWithSemicolons
      case _ => throw new UnknownError("Unknown file format: " + format)
    }
    implicit val typeInfo: TypeInformation[LabeledData] = TypeInformation.of(classOf[LabeledData])
    sc.readTextFile(input).map { line => parse(line, maxDim, negY) }
  }

  def parseLibSVM(line: String, maxDim: Int, negY: Boolean = true): LabeledData = {
    val splits = line.trim.split(" ")
    if (splits.length < 1)
      return null

    var y = splits(0).toDouble
    if (negY && Math.abs(y - 1) > Maths.EPS)
      y = -1

    val nnz = splits.length - 1
    val indices = new Array[Int](nnz)
    val values = new Array[Double](nnz)
    for (i <- 0 until nnz) {
      val kv = splits(i + 1).trim.split(":")
      indices(i) = kv(0).toInt-1
      values(i) = kv(1).toDouble
    }
    val x = SparseVector(maxDim, indices, values)

    LabeledData(y, x)
  }

  def parseLibSVMWithSemicolons(line: String, maxDim: Int, negY: Boolean = true): LabeledData = {
    val splits = line.trim.split(" ")
    if (splits.length < 1)
      return null

    var y = splits(0).toDouble
    if (negY && Math.abs(y - 1) > Maths.EPS)
      y = -1

    val nnz = splits.length - 1
    val indices = new Array[Int](nnz)
    val values = new Array[Double](nnz)
    for (i <- 0 until nnz) {
      val kv = splits(i + 1).trim.split(";")
      indices(i) = kv(0).toInt-1
      values(i) = kv(1).toDouble
    }
    val x = SparseVector(maxDim, indices, values)

    LabeledData(y, x)
  }

  def parseCSV(line: String, maxDim: Int, negY: Boolean = true): LabeledData = {
    val splits = line.trim.split(",")
    if (splits.length < 1)
      return null

    var y = splits(0).toDouble
    if (negY && Math.abs(y - 1) > Maths.EPS)
      y = -1

    val nnz = splits.length - 1
    val values = splits.slice(1, nnz + 1).map(_.trim.toDouble)
    val x = DenseVector(values)

    LabeledData(y, x)
  }

  def parseDummy(line: String, maxDim: Int, negY: Boolean = true): LabeledData = {
    val splits = line.trim.split(",")
    if (splits.length < 1)
      return null

    var y = splits(0).toDouble
    if (negY && Math.abs(y - 1) > Maths.EPS)
      y = -1

    val nnz = splits.length - 1
    val indices = splits.slice(1, nnz + 1).map(_.trim.toInt)
    val values = Array.fill(nnz)(1.0)
    val x = SparseVector(maxDim, indices, values)

    LabeledData(y, x)
  }

}
