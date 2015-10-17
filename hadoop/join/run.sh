#!/bin/bash

set -x

mvn package -Dmaven.test.skip=true -f pom.xml
[ $? -ne 0 ] && echo "compile error"

rm -rf data/output
hadoop jar target/join.jar com.nowcoder.course.Join data/input1 data/input2 data/output
[ $? -ne 0 ] && echo "excecute error"

exit 0
