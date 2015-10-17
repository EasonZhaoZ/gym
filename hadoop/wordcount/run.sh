#!/bin/bash

set -x

mvn package -Dmaven.test.skip=true -f pom.xml
[ $? -ne 0 ] && echo "compile error"

rm -rf data/output
hadoop jar target/wordcount.jar com.nowcoder.course.WordCount data/input data/output
[ $? -ne 0 ] && echo "excecute error"

exit 0
