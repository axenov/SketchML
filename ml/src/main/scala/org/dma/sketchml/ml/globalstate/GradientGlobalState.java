package org.dma.sketchml.ml.globalstate;

import org.dma.sketchml.ml.gradient.Gradient;

/**
 * Interface to maintain a global gradient in a distributed
 * machine learning system.
 */
public interface GradientGlobalState {

  /**
   * Initialize all the resources needed to
   * calculate a global gradient.
   */
  void init();

  /**
   * Sent a local gradient value.
   * @param localGradient local value
   */
  void sentLocalGradient(Gradient localGradient);

  /**
   * @return global gradient value
   */
  Gradient getGlobalGradient() throws InterruptedException;
}
