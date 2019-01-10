package org.dma.sketchml.ml.data

import org.apache.flink.ml.math.Vector

case class LabeledData(label: Double, feature: Vector) {

  override def toString: String = s"($label $feature)"
}
