# Advanced AI Repository

## Overview

This repository contains the work and models for the Advanced AI module, which focuses on the resolution of 2 common problems facing E commerce platforms 

* Product Reorder Prediction – predicting which items a customer is likely to reorder based on historical behaviour.
* Freshness Classification – classifying images of produce as either *fresh* or *rotten*.

Each model is developed and documented independently, with explanations provided in their respective subdirectories.

---

## Structure

```
Advanced-AI/
│
├── Prototyping/   
│
└── Modelling/
    │
    ├── ProdPred/
    │   └── LogModel/
    │   └── Dataset/
    │
    └── RottenFresh/
        
```

### Folder Descriptions

* **Prototyping/**
  Contains early-stage model development. This area is used to test ideas before implementation.

* **Modelling/**
  Contains the finalised implementations of the two main systems:

  * **ProdPred/**
    Focuses on predicting products that a customer will reorder, using linear models and collaborative filtering.

  * **RottenFresh/**
    Focuses on image classification, distinguishing between fresh and rotten produce and identifying problem areas.

Each modelling subfolder includes its own dedicated README 

---

## Getting Started

To explore the project:

1. Start with the `Modelling/` directory for the final implementations.
2. Refer to the individual READMEs in:

   * `ProdPred/`
   * `RottenFresh/`
3. Use the `Prototyping/` folder to understand the development process and experimentation.

