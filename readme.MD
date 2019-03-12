## Introduction
This is the fork of [SketchML](https://github.com/ccchengff/SketchML) which is a [Apache Spark](https://spark.apache.org/), batch implementation of distributed machine learning
with efficient sketch-based gradient compression. Our version incorporates [Apache Flink](https://flink.apache.org/) and its Streaming API to make use of mentioned compression
in online machine learning scenario.

This project was developed during BDAPRO (Big Data Analytics Project) course given by [DIMA group at TU Berlin](https://www.dima.tu-berlin.de/).

## Overview of changes to original version
The whole Spark pipeline is completely replaced with Flink + Parameter Server one. All Spark-based data structures were replaced with Flink's equivalents 
and slightly adjusted in case of differently named or missing methods. Moreover [Parser](./ml/src/main/scala/org/dma/sketchml/ml/data/Parser.scala) was
redesigned to be able to read the data into streams.

Project incorporates [Flink parameter server library](https://github.com/FlinkML/flink-parameter-server) and whole 
streaming machine learning pipeline is based on it.
Parameter server `transform` method, parameter initialization logic and parameter update logic are defined in 
[GeneralizedLinearModel](./ml/src/main/scala/org/dma/sketchml/ml/algorithm/GeneralizedLinearModel.scala).
Worker logic is defined in 
[GradientDistributionWorker](./ml/src/main/scala/org/dma/sketchml/ml/parameterserver/GradientDistributionWorker.scala).

In short streaming pipeline looks like this:
1. Split incoming data into windows. 
1. When worker receives new window it stores it in local FIFO queue and sends pull request to parameter server.
1. Parameter server answers worker with current value of the model (weights of the function).
1. Worker based on the current model and first window from the FIFO queue computes new gradient as well as validates the model.
1. Worker compresses the gradient using SketchML algorithm.
1. Worker pushes compressed gradient to parameter server.
1. Parameter server decompresses the gradient and updates the model.

The whole process continues infinitely regardless of the input file size, so after every window is processed
it is necessary to cancel Flink job manually. The end of the processing can be inferred from the logs.

## How to run the project

### Versions of used software
* Java 1.8
* Maven 3.5.4
* Flink 1.7.0
* Scala 2.11

### Executable jar creation
Project is dependent on parameter server library.
```xml
<dependency>
    <groupId>hu.sztaki.ilab</groupId>
    <artifactId>flink-ps_2.11</artifactId>
    <version>0.1.0</version>
</dependency>
```
However it is not available in remote repositories, that's why it is necessary to compile [Flink parameter server library](https://github.com/FlinkML/flink-parameter-server)
to `.jar` format and then publish it to local [Maven](https://maven.apache.org/) repository (jar should be effectively placed under `repositories/hu/sztaki/ilab/flink-ps_2.11/0.1.0` path).

Then to create executable jar of the whole project it's enough to run `mvn package` from the root folder. Executable jar, which can be used to 
submit on Flink will be stored under `.ml/target/ml-1.0.0-jar-with-dependencies.jar`

### Must have parameters
To run the program, you should pass AT LEAST these parameters:

`--flink.sketchml.input.path file://path_to_file`
 
`--flink.sketchml.worker.num number_of_workers` - number of workers which communicate with parameter server (this should be equal to Flink parallelism)
 
`--flink.sketchml.feature.num number_of_features` - number of features in your input file
 
`--flink.sketchml.algo (LogisticRegression|LinearRegression|SupporVectorMachine`
 
`--flink.sketchml.input.format (csv|libsvm)`

PROTIP: remember that path to the file is in URI format, so in Windows it looks unusual e.g.:

`--flink.sketchml.input.path file:///C:/main_folder/data_file`



### Additional parameters
There are additional parameters, which have some default values, but are possible to change if you want to conduct different experiments.

Template:

`param.name default_value`

#### Streaming parameters

Number of elements in one window of data passed to worker (**to find good value of window size experiments on given data are needed**)

`--flink.sketchml.window.size 100`

#### Machine learning parameters
Learning rate used in model updates:

`--flink.sketchml.learn.rate 0.1` 

Lambda value used in loss functions:

`--flink.sketchml.reg.l2 0.1`

#### Sketch parameters
Parameters below regard sketch compression configuration. To understand their full meaning please read [original SketchML paper](http://net.pku.edu.cn/~yangtong/uploads/SketchML.pdf).
  
`--flink.sketchml.gradient.compressor GRADIENT_COMPRESSOR_SKETCH`

`--flink.sketchml.quantization.bin.num 256`

`--flink.sketchml.minmaxsketch.group.num 8`

`--flink.sketchml.minmaxsketch.row.num 2`

`--flink.sketchml.minmaxsketch.col.ratio 0.3`

`--flink.sketchml.fixed.point.bit.num 8`