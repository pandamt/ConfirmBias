# Confirmation Bias on Human Decison-making

This repository contains model simulation code for both training and test conditions.

## functionsEstimator_v4.m
This file contains a set of Bayes estimators with neural signal nodes for different conditions.

## functionSub.m
This file contains sub-functions (e.g. linear regression, gaussian filter).

## ConfirmBias_Test_v4.m
This file executes the Bayes estimator for two test conditions (categorical judgement/attentional cue).

## Executation Parameters
- flagMemDecay(1): memory decay on the Mu level during the stimulus presentation period, 0 = no decay, 1 =  decay
- flagMemDecay(2): memory decay on the C level during the stimulus presentation period, 0 = no decay, 1 = decay
- flagMemDecay_break(1): memory decay on the Mu level during the intermediate break, 0 = no decay, 1 =  decay
- flagMemDecay_break(2): memory decay on the C level during the intermediate break, 0 = no decay, 1 = decay
- rateDecayC: memory decay rate for C
- sigma, truncate: memory decay parameters for Mu (Gaussian filter)
- sigmaMotor: motor noise parameter
- rateConsol: proportional consolidation rate during the intermediate break in the categorical judgement condition
- leak_period: length of memory leak period during the intermediate break
