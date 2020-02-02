* Project path: ...

├── README.md                  <- This file
│
├── data
│   ├── intermediate           <- Intermediate data (clean, temporary data)
│   ├── output                 <- Model results and scoring
│   ├── processed              <- The final data sets for modeling
│   └── raw                    <- The original, immutable data dump
│
├── models                     <- Trained and serialized models
│
├── notebooks                  <- Jupyter notebooks
│
├── references                 <- Data explanatory materials
│
├── reports                    <- Generated analysis as HTML, PDF etc.
│   └── figures                <- Generated charts and figures for reporting
│
├── requirements.yml           <- Requirements file for conda environment
│
├── src                        <- Source code for use in this project.
    │        
    ├── tests                  <- Automated tests to check source code
    │    
    ├── data_preparation       <- Source code to generate data
    │
    ├── features_extraction    <- Source code to extract and create features
    │
    ├── modelling              <- Source code to train and score models
    │
    └── visualization          <- Source code to create visualizations
    
    
* Note 1: The idea is mainly drawn from https://towardsdatascience.com/lessons-from-a-real-machine-learning-project-part-1-from-jupyter-to-luigi-bdfd0b050ca5
* Note 2: For futher reading:
    [1] https://drivendata.github.io/cookiecutter-data-science/
    [2] https://towardsdatascience.com/manage-your-data-science-project-structure-in-early-stage-95f91d4d0600

