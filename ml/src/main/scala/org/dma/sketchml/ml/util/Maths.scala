package org.dma.sketchml.ml.util

import org.apache.flink.ml.math.{DenseVector, SparseVector, Vector}

import scala.collection.mutable.ArrayBuffer
import scala.math.sqrt

@SerialVersionUID(1L)
object Maths extends Serializable {
  val EPS = 1e-8

  def secondNorm(data: Vector): Double = {
    sqrt(data.map(x => x._2 * x._2).sum)
  }

  def add(k1: Array[Int], v1: Array[Double], k2: Array[Int],
          v2: Array[Double]): (Array[Int], Array[Double]) = {
    val k = ArrayBuffer[Int]()
    val v = ArrayBuffer[Double]()
    var i = 0
    var j = 0
    while (i < k1.length && j < k2.length) {
      if (k1(i) < k2(j)) {
        k += k1(i)
        v += v1(i)
        i += 1
      } else if (k1(i) > k2(j)) {
        k += k2(j)
        v += v2(j)
        j += 1
      } else {
        k += k1(i)
        v += v1(i) + v2(j)
        i += 1
        j += 1
      }
    }
    (k.toArray, v.toArray)
  }

  def dot(a: Vector, b: Vector): Double = {
    (a, b) match {
      case (a: DenseVector, b: DenseVector) => dot(a, b)
      case (a: DenseVector, b: SparseVector) => dot(a, b)
      case (a: SparseVector, b: DenseVector) => dot(a, b)
      case (a: SparseVector, b: SparseVector) => dot(a, b)
    }
  }

  def dot(a: DenseVector, b: DenseVector): Double = {
    require(a.size == b.size, s"Dot between vectors of size ${a.size} and ${b.size}")
    //(a.values, b.values).zipped.map(_*_).sum
    val size = a.size
    val aValues = a.data
    val bValues = b.data
    var dot = 0.0
    for (i <- 0 until size) {
      dot += aValues(i) * bValues(i)
    }
    dot
  }

  def dot(a: DenseVector, b: SparseVector): Double = {
    require(a.size == b.size, s"Dot between vectors of size ${a.size} and ${b.size}")
    val aValues = a.data
    val bIndices = b.indices
    val bValues = b.data

    //val size = b.numActives
    val size = bIndices.length
    var dot = 0.0
    for (i <- 0 until size) {
      val ind = bIndices(i)
      dot += aValues(ind) * bValues(i)
    }
    dot
  }

  def dot(a: SparseVector, b: DenseVector): Double = dot(b, a)

  def dot(a: SparseVector, b: SparseVector): Double = {
    require(a.size == b.size, s"Dot between vectors of size ${a.size} and ${b.size}")
    val aIndices = a.indices
    val aValues = a.data
    //val aNumActives = a.numActives
    val aNumActives = a.indices.length
    val bIndices = b.indices
    val bValues = b.data
    //val bNumActives = b.numActives
    val bNumActives = b.indices.length
    var aOff = 0
    var bOff = 0
    var dot = 0.0
    while (aOff < aNumActives && bOff < bNumActives) {
      if (aIndices(aOff) < bIndices(bOff)) {
        aIndices(aOff) += 1
      } else if (aIndices(aOff) > bIndices(bOff)) {
        bOff += 1
      } else {
        dot += aValues(aOff) * bValues(bOff)
        aOff += 1
        bOff += 1
      }
    }
    dot
  }

  def euclidean(a: Array[Double], b: Array[Double]): Double = {
    require(a.length == b.length)
    (a, b).zipped.map((x, y) => (x - y) * (x - y)).sum
  }

  def cosine(a: Array[Double], b: Array[Double]): Double = {
    val va = new DenseVector(a)
    val vb = new DenseVector(b)
    //dot(va, vb) / (Vectors.norm(va, 2) * Vectors.norm(vb, 2))

    dot(va, vb) / (this.secondNorm(va) * this.secondNorm(vb))

  }

}
