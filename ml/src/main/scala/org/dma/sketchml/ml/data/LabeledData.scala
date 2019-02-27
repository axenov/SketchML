package org.dma.sketchml.ml.data

import org.apache.flink.ml.math.Vector

@SerialVersionUID(1L)
case class LabeledData(label: Double, feature: Vector) extends Serializable {

  override def toString: String = s"($label $feature)"
}
