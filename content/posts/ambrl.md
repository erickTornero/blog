---
title: "Advances in Model Based Reinforcement Learning"
date: 2020-01-30T15:39:27-05:00
draft: false
math: true
markup: mmark
---

Introduction to the basics on model-based reinforcement learning and the recent advances on this topic.

<!--more-->
## Introduction


Model-based reinforcement learning, can be understanding as a learning the transition function $$P(s', r|s,a)$$ in the markov desicion process (MDP). Then this model can be used for planning to take actions which in intermediate states the value function is computed, as can be seen in **Figure 1**, in the model-learning branch. Direct RL, means model-free RL where the policy or value-function is computed directly from experience.

{{<figure src="https://ericktornero.github.io/blog/images/squeme_mbmf.png" title="Figure 1, Taken from [1]">}}


Basic form of model-based reinforcement learning can be seen in **Dynamic Programming**, in which is assumed a prior knowladge over the dynamics or the transition function. Hoewever, in real world, the dynamics is usually unknown and can be very complex to model. For these kind of problems, model learning can be used just as supervised learning.

{{<figure src="https://ericktornero.github.io/blog/images/gridworld_hchetaah.png" title="Figure 2, Taken from [1]">}}

For low dimensional state-action space, Gaussian Process (GPs) can be used to approximate the transition function. However when complexity in the model increasses, e.g. in robotics control, gaussian process used to be inadequate. Neural Networks however are known by its high adaptavility to complex functions as in images, and in recent years, has beend showed interesting results in several applications, in that sense, this post focused in recent advances in MBRL that uses Neural Networks for the approximation of the transition function.

## Basic concepts in Model-Based Reinforcement Learnig



## Deep Reinforcement Learning with a handful of trials with probabilistic models

This is a resume of paper published in NeuriPS 2018 Montreal, we create a brief summary and the highlights of this paper

This paper introduces uncertainty-aware to the dynamics model. In comparison

$$TD_x = e^2$$
