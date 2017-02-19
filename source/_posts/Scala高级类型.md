---
title: Scala高级类型
date: 2016-12-07 08:50:00
updated	: 2016-12-07 08:50:00
permalink: scala
tags:
- Scala
- Spark

categories:
- language
- scala
---


###　Scala类型系统总结
类型	语法类型
 类	class Person
 特质	trait Closable
 元组类型	(T1,T2,T3,…)
 函数类型	(T1,T2,t3,…)=>T
 参数类型（泛型）	class Person[T1,T2,…]
 单例类型	this.type
 类型投影	Outter#Inner
 复合类型	A with B with C…
 结构体类型	{def f():Unit ….}
 中置类型	T1 A T2
 存在类型	T forSome {}

 ### Reference
［１］ https://yq.aliyun.com/articles/60370?spm=5176.8251999.569296.24
