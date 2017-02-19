
利用 checkpoint 来保留上一个窗口的状态，这样可以做到移动窗口的更新统计。

首先还是先演示一下 spark 里传回调函数的用法，上一篇里用 DStream 处理模拟了 SUM()，这个纯加法是最简单的了，那么如果 AVG() 怎么做呢？
```scala
val r = logs.filter(l => l.path.equals("/var/log/system.log")).filter(l => l.lineno > 70)
    r.map(l => l.message -> (l.lineno, 1)).reduceByKey((a, b) => {
      (a._1 + b._1, a._2 + b._2)
    }).map(t => AlertMsg(t._1, t._2._2, t._2._1/t._2._2)).print()
```
这段跟之前做 SUM 的那段的区别：

DStream 处理成 PairDStream 的时候，Value 不是单纯的 1，而是一个 Seq[Double, Int]。避免了上一个示例里分开两个 DStream 然后再 join 起来的操作；
给 reduceByKey 传了一个稍微复杂的匿名函数。在这一个函数里计算了 SUM 和 COUNT，后面 map 只需要做一下除法就是 AVG 了。
不过这里还用不上上一次窗口的状态。真正需要上一次窗口状态的，是 reduceByKeyAndWindow 和 updateStateByKey。reduceByKeyAndWindow 和 reduceByKey 的区别，就是除了计算新数据的函数，还要传递一个处理过期数据的函数。

下面用 updateStateByKey ，演示一下如何计算每个窗口的平均值，跟上一个窗口的平均值的涨跌幅度，如果波动超过 10%，则输出：

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.StreamingContext._
import scala.util.parsing.json.JSON
object LogStash {
  case class LogStashV1(message:String, path:String, host:String, lineno:Double, timestamp:String)
  case class Status(sum:Double = 0.0, count:Int = 0) {
    val avg = sum / scala.math.max(count, 1)
    var countTrend = 0.0
    var avgTrend = 0.0
    def +(sum:Double, count:Int): Status = {
      val newStatus = Status(sum, count)
      if (this.count > 0 ) {
        newStatus.countTrend = (count - this.count).toDouble / this.count
      }
      if (this.avg > 0 ) {
        newStatus.avgTrend = (newStatus.avg - this.avg) / this.avg
      }
      newStatus
    }
    override def toString = {
      s"Trend($count, $sum, $avg, $countTrend, $avgTrend)"
    }
  }
  def updatestatefunc(newValue: Seq[(Double, Int)], oldValue: Option[Status]): Option[Status] = {
    val prev = oldValue.getOrElse(Status())
    var current = prev + ( newValue.map(_._1).sum, newValue.map(_._2).sum )
    Some(current)
  }
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("LogStash")
    val sc  = new SparkContext(sparkConf)
    val ssc = new StreamingContext(sc, Seconds(10))
    val lines = ssc.socketTextStream("localhost", 8888)
    val jsonf = lines.map(JSON.parseFull(_)).map(_.get.asInstanceOf[scala.collection.immutable.Map[String, Any]])
    val logs = jsonf.map(data => LogStashV1(data("message").toString, data("path").toString, data("host").toString, data("lineno").toString.toDouble, data("@timestamp").toString))
    val r = logs.filter(l => l.path.equals("/var/log/system.log")).filter(l => l.lineno > 70)
    r.map(l => l.message -> (l.lineno, 1)).reduceByKey((a, b) => {
      (a._1 + b._1, a._2 + b._2)
    }).updateStateByKey(updatestatefunc).filter(t => t._2.avgTrend.abs > 0.1).print()
    ssc.start()
    ssc.awaitTermination()
  }
}
这里因为流数据只有 sum 和 count，但是又想留存两个 trend 数据，所以使用了一个新的 cast class，把 trend 数据作为 class 的 value member。对于 state 来说，看到的就是一整个 class 了。

依然有参考资料：

http://blog.cloudera.com/blog/2014/11/how-to-do-near-real-time-sessionization-with-spark-streaming-and-apache-hadoop/
http://www.scottlogic.com/blog/2013/07/29/spark-stream-analysis.html
https://github.com/rshepherd/spark-streaming-average/blob/master/src/main/scala/StreamingAverage.scala
