package org.dma.sketchml.ml.globalstate;

import io.atomix.cluster.messaging.ClusterEventService;
import io.atomix.cluster.messaging.Subscription;
import io.atomix.core.Atomix;
import io.atomix.core.AtomixBuilder;
import io.atomix.core.barrier.DistributedCyclicBarrier;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.dma.sketchml.ml.conf.MLConf;
import org.dma.sketchml.ml.gradient.Gradient;

@Deprecated
public class AtomixGradientGlobalState implements GradientGlobalState {
  private static final String GRADIENT_BARRIER = "GRADIENT_BARRIER";
  private static final String GRADIENT = "GRADIENT_MESSAGE";

  //Net
  private MLConf mlConf;
  private Atomix atomix;
  private AtomixBuilder builder = Atomix.builder();
  private DistributedCyclicBarrier barrier;
  private ClusterEventService eventService;
  private Subscription subscription;

  //Gradient
  private AtomicReference<Gradient> localGradient = new AtomicReference<>();
  private Gradient[] gradientsSum;
  private AtomicInteger gradientsReceived = new AtomicInteger(0);
  private DirectExecutor executor = new DirectExecutor();

  public AtomixGradientGlobalState(MLConf mlConf) {
    this.mlConf = mlConf;
  }

  public AtomixGradientGlobalState(MLConf mlConf, AtomixBuilder builder) {
    this.mlConf = mlConf;
    this.builder = builder;
  }

  class DirectExecutor implements Executor {
    public void execute(Runnable r) {
      r.run();
    }
  }

  @Override
  public void init() {
    atomix = this.builder.build();
    atomix.start().join();
    this.barrier = atomix.getCyclicBarrier(GRADIENT_BARRIER);
    this.eventService = atomix.getEventService();
    this.gradientsSum = new Gradient[mlConf.workerNum()];
    this.subscribe();
  }

  private void subscribe() {
    this.subscription = eventService.subscribe(GRADIENT, remoteGradient -> {
      Gradient gradient = (Gradient) remoteGradient;
      this.addGradient(gradient);
    }, executor).join();
  }

  public void stopGradient(){
    subscription.close();
    atomix.stop();
  }

  @Override
  public void sentLocalGradient(Gradient localGradient) {
    Gradient compressedGradient = Gradient.compress(localGradient, mlConf);
    this.eventService.send(GRADIENT, compressedGradient);
    this.addGradient(localGradient);
  }

  private synchronized void addGradient(Gradient gradient){
    this.gradientsSum[this.gradientsReceived.getAndIncrement()] = gradient;
  }

  @Override
  public synchronized Gradient getGlobalGradient() throws InterruptedException {
    this.barrier.wait();
    this.gradientsReceived.set(0);
    Gradient newGradient = Gradient.sum(mlConf.featureNum(), gradientsSum);
    newGradient = Gradient.compress(newGradient, mlConf);
    newGradient.timesBy(1.0/mlConf.workerNum());
    localGradient.set(newGradient);
    return localGradient.get();
  }
}
