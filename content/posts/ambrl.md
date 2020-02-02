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


Model-based reinforcement learning is a branch in Reinforcement Learning where the transition function $$P(s', r|s, a)$$ in the Markov decision process (MDP) is used. This model is used to improve or create the policy (planning), where the value function is calculated as an intermediate step, as can be seen in **Figure 1** in the *model-learning* branch. *Direct RL*, means model-free RL where the policy or value-function is computed *directly* from experience and it is not needed to model the dynamics. However, these kinds of methods are too poor sample efficient due to that these methods rely on the probability to find good interactions, this becomes even more complicated when the environment has high dimensional state-action space and when sparse rewards are presented.

{{<figure src="https://ericktornero.github.io/blog/images/squeme_mbmf.png" title="Figure 1, Taken from [1]">}}


The basic form of model-based reinforcement learning can be seen in **Dynamic Programming**, which is assumed a prior knowledge over the dynamics or the transition function. However, in the real world, the dynamics are usually unknown and can be very complex to model. For these kinds of problems, model learning can be used just as supervised learning. In **Fig. 2**, the left picture shows a Simple Gridworld example with a discrete state and actions. In this case the transition function is known, for example $$P(s_2|s_1,right)=1.0$$ and $$P(s_2|s_1, left)=0.0$$, where $$s_x$$: represents the slot $$x \in \{0, 15\}$$. In the right picture, Halfcheetah in the [Mujoco Environment](http://www.mujoco.org/) is shown, where a priory of the dynamics of this environment is unknown and complex. 

{{<figure src="https://ericktornero.github.io/blog/images/gridworld_hchetaah.png" title="Figure 2, left: Gridworld environment, taken from [1]. Right: Halfcheetah Environment">}}

For low dimensional state-action space, Gaussian Process (GPs) can be used to approximate the transition function. However when complexity in the model increasses, e.g. in robotics control, gaussian process used to be inadequate. Neural Networks however are known by its high adaptavility to complex functions, and in recent years, has beend showed interesting results in several applications, such as image classification. In that sense, this post focused in recent advances in MBRL that uses Neural Networks for the approximation of the transition function.

## Basic concepts in Model-Based Reinforcement Learnig

Reinforcement learning Framework is defined over a Markov Decision Process, and elements in an MDP are defined in the tuple $$(\mathcal{S}, \mathcal{A}, \mathcal{R}, \mathcal{P}, \gamma)$$, where $$\mathcal{S}$$ is the state space, $$\mathcal{A}$$ is the action space, $$\mathcal{R}$$ is the reward and $$\mathcal{P(s', r|s, a)}$$ define the transition function. In **Fig. 3**, the interaction interaction agent-envrionment is shown, given an observed state $$s_t \in \mathcal{S}$$ at any particular timestep $$t$$, the agent take an action $$a_t \in \mathcal{A}$$ drawn from its policy $$\pi(a_t|s_t)$$. The environment respond with the next state $$s_{t+1}$$ and the reward $$r_{t+1}$$ produced by take action $$a_t$$. 

{{<figure src="https://ericktornero.github.io/blog/images/mdp.png" title="Figure 3, Interaction in MDP">}}

Either model-based or model-free methods try to compute a **policy** $$\pi(a|s)$$ that maximizes the expected reward. In the case of mode-based, the policy is computed indirectly by using the transition model $$\mathcal{P}(s',r|s, a)$$ for planning. As an example, some algorithms are explaining in the following section called: **Dynamic Programming**.

### Dynamic Programming ###

Dynamic programming are sets of algorithms where is assumed that the environment transition function is priorly known. The aim of RL is summarized in the following equation:

$$\mathbb{E}[G_t] = \mathbb{E}_{\pi}[\sum_t \gamma^tr_t | s_t,a_t]$$

$$\mathbb{E}[G_t] = \sum_s [\sum_t v(s) | a_t]$$

$$v_\pi(s) = \sum_{s, r} \sum_{a} p(s'|s,a) \pi(a|s)[r + v(s)]$$



## Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning
### Anusha Nagabandi et al. UC Berkeley 2017

## Deep Reinforcement Learning with a handful of trials with probabilistic models

This is a resume of paper published in NeuriPS 2018 Montreal, we create a brief summary and the highlights of this paper

This paper introduces uncertainty-aware to the dynamics model. In comparison


## Deep Dynamics Models for Learning Dexterous Manipulation
### Anusha Nagabandi et al. UC Berkeley 2019

## Learning Latent Dynamics for Planning from Pixels

