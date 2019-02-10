To run the program, you should pass AT LEAST these parameters:

`--flink.sketchml.input.path file://path_to_file`
 
`--flink.sketchml.worker.num number_of_workers`
 
`--flink.sketchml.feature.num number_of_features`
 
`--flink.sketchml.algo LogisticRegression`
 
`--flink.sketchml.input.format (csv|libsvm|dummy|libsvmSemicolons)`

PROTIP: remember that path to the file is in URI format, so on Windows it looks quite funny e.g.:

`--flink.sketchml.input.path file:///C:/main_folder/data.txt`

Example configuration, which I used to run the program:

`--flink.sketchml.input.path file:///C:/Projects/SketchML/ml/src/main/resources/abalone.txt --flink.sketchml.worker.num 1 --flink.sketchml.feature.num 7 --flink.sketchml.algo LogisticRegression --flink.sketchml.input.format csv --flink.sketchml.window.size 100`