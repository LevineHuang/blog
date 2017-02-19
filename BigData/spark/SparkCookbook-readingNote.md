## 5 Spark Streaming
---

#### Word count using Streaming

1.  Start the Spark shell and give it some extra memory:
```sh
$ spark-shell --driver-memory 1G
```
2.  Stream specific imports:
```scala

import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds,StreamingContext}
import org.apache.spark.storage.StorageLevel
import StorageLevel._

```
3.  Import for an implicit conversion:
import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
4.  Create StreamingContext with a 2 second batch interval:
scala> val ssc = new StreamingContext(sc, Seconds(2))
5.  Create a SocketTextStream Dstream on localhost with port 8585 with the
MEMORY_ONLY caching:
scala> val lines = ssc.socketTextStream("localhost",8585,MEMORY_ONLY)
6.  Divide the lines into multiple words:
scala> val wordsFlatMap = lines.flatMap(_.split(" "))
7.  Convert word to (word,1), that is, output 1 as the value for each occurrence of a word
as the key:
scala> val wordsMap = wordsFlatMap.map( w => (w,1))

8.  Use the reduceByKey method to add a number of occurrences for each word as the
key (the function works on two consecutive values at a time, represented by a and b):
scala> val wordCount = wordsMap.reduceByKey( (a,b) => (a+b))
9.  Print wordCount:
scala> wordCount.print
10.  Start StreamingContext; remember, nothing happens until StreamingContext
is started:
scala> ssc.start
11.  Now, in a separate window, start the netcat server:
$ nc -lk 8585
12. Enter different lines, such as to be or not to be:
to be or not to be
13. Check the Spark shell, and you will see word count results like the following
screenshot:
```
