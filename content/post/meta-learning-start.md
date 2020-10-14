---
title: "Introduction to Meta Learning"
date: 2020-04-27T02:03:11-05:00

draft: false
math: true
markup: mmark
tags: ["meta-learning", "Artificial Intelligence"]
categories: ["blogpost"]
author: "Erick Tornero"

---

Meta-learning is one of the hottest topics of research. In recent years, many researchers have taken interested in meta-learning due to its promise to reach *generalization*. In this blog post, we present a brief introduction to concepts in meta-learning and some applications which will immerse you in this wonderful field.

## An starting point of view

One can notice that humans or animals can learn a task quickly when they have prior knowledge on another similar task. For instance, driving a motorcycle is an esier task when before you know how to ride a bike, or how babies start to walk, using its knowledge in how to stand up. Naive machine learning techniques usually are not able to do things like that, if we see common problems like classification, regression or reinforcement learning, usually our model can solve just an specific problem, it is a problem in how to generalize even to minor changes in a unique environment. So, meta-larning is a field which try to mitigate this problem, we can define: *meta-learning as one of the fields that try to take advantage of prior knowledge of many tasks to learn as quickly as possible a related new task.*


{{<figure src="https://ericktornero.github.io/blog/images/metalearn/babywalk.gif" caption="**Figure 1**, Babies can take advantage of its knowledge in how to stand-up to start to walk">}}


## Relation and differences with related topics

Ok, the previous definition can be a bit confusing to other related fields such as transfer learning, continual learning, multi-task learning, among others, because all of them rely on the laverage of previous knowladge from related taks. But, think that in meta-learning a meta-representation is explicitly learned. 

For instance, in transfer learning, parameters trained with an specific task are transfer to another related task. Here, there's not any optimal representation to be transfered over tasks, don't matters if the parameters are the optimal to be transfered over different tasks. Or in Continual learning where adaptation to new possible non-stationary distribution is doing in continual way by laveraging previous knowledge, but there's not necessary the existence of a meta-representation. While for at least these methods are differents to meta-learning in the escense, meta-learning can be used to reach those purpose.



[//]: <> (It is common to get confused with other topics that are similar to meta-learning as transfer learning, continual learning, or multitask learning. Here we explain the main differences.)

[//]: <> (**Transfer Learning:** Transfer learning aims to learn a task by taking advantage of a previous related learned task by transferring its parameters. We can notice that this is very similar to the meta-learning target, however, in meta-learning, it is explicitly defined the notion of generalization over tasks, while in transfer learning it is not necessary. Another difference is that while meta-learning techniques can transfer parameters, it is not just limited to this, we will see a further explanation later.)

[//]: <> (**Continual Learning:** Continual learning aims to lifelong learning, adapting to possible changes in the environment trying to avoid catastrophic forgetfulness. )

[//]: <> (As well as meta-learning, continual learning takes advantage of the previously acquired knowledge to adapt faster along time. Continual learning can use meta-learning for its purpose but is not limited to do it. Here, like in transfer learning, there's not explicit meta-optimization over tasks.)

[//]: <> (**Multi-task Learning:** Multi-task learning aims to achieve good performance over a set of tasks. Usually Multi-task learning promotes good generalization)

## Common approaches of Meta-learning

There's lot of approaches to tackle the meta-learning problem, here we discuss the two most common, main difference is how they considers the meta-representation: Optimization based methods and Black Box methods.

### Optimization based methods

### Black Box methods


## Some applications of Meta-Learning

Applications of meta-learning are diverece for superised learning, unsupervised learning, reinforcement learning, etc. Here we discuss some of the most important configurations.

### Few-shot learning

### Reinforcement learning
