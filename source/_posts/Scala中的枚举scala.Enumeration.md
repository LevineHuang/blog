---
title: Scala中的枚举
date: 2016-12-07 08:30:00
updated	: 2016-12-07 08:30:00
permalink: scala
tags:
- Scala
- Spark

categories:
- language
- scala
---

### scala枚举简介

在Scala中并没有枚举类型，但在标准类库中提供了Enumeration类来产出枚举。扩展Enumeration类后，调用Value方法来初始化枚举中的可能值。

内部类Value实际上是一个抽象类，真正创建的是Val。因为实际上是Val，所以可以为Value传入id和name

如果不指定，id就是在前一个枚举值id上加一，name则是字段名

###　scala枚举示例
```scala
object TrafficLightColor extends Enumeration {
  type TrafficLightColor = Value
  val Red = Value(0, "Stop")
  val Yellow = Value(10)
  val Green = Value("Go")
}

object Margin extends Enumeration {
  type Margin = Value
  val TOP, BOTTOM, LEFT, RIGHT = Value
}

import test.TrafficLightColor._
import test.Margin._
object Driver extends App {
  println(BOTTOM, BOTTOM.id)

  def doWhat(color: TrafficLightColor) = {
    if (color == Red) "stop"
    else if (color == Yellow) "hurry up" else "go"
  }

  //使用match匹配
  def doWhat2(color: TrafficLightColor) = color match {
    case Red    => "stop"
    case Yellow => "hurry up"
    case _      => "go"
  }

  // load Red
  val red = TrafficLightColor(0) // Calls Enumeration.apply 
  println(red, red.id)
  println(doWhat(red))
  println(doWhat2(TrafficLightColor.Yellow))
  
   //打印出所有枚举
  Margin.values.foreach { v => println(v,v.id)}
}
```

```scala
object Main extends Application {
    object WeekDay extends Enumeration {
      type WeekDay = Value
      val Mon, Tue, Wed, Thu, Fri, Sat, Sun = Value
    }

    import WeekDay._

    def isWorkingDay(d: WeekDay) = ! (d == Sat || d == Sun)

    WeekDay filter isWorkingDay foreach println
  }
```

### Reference
https://my.oschina.net/cloudcoder/blog/490061
