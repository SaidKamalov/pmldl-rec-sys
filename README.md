# pmldl-rec-sys
Code repository for assignment2:  Movie Recommender System.
<p>PMLDL course of Innopolis university.</p>

### Repository structure:
```
movie-recommender-system
├── README.md               # The top-level README
│
├── data
│   └── raw                 # The original, immutable data
│
├── models                  # Trained and serialized models, final checkpoints
│
├── notebooks               #  Jupyter notebooks. Naming convention is a number (for ordering),
│                               and a short delimited description, e.g.
│                               "1.0-initial-data-exporation.ipynb"            
│ 
├── references 
|   └── references.md             # manuals, useful links, and all other explanatory materials.
│
├── reports
│   ├── figures             # Generated graphics and figures to be used in reporting
│   └── final_report.pdf    # Report containing data exploration, solution exploration, training process, and evaluation
│
└── benchmark
    ├── data                # dataset used for evaluation 
    └── evaluate.py         # script that performs evaluation of the given model
```

### Get the suggestions
To get movie suggestions for a specific user you just need to run the ```get_recommendations.py```
```
python get_recommendations.py --user_id [id] --num_of_reqs[number of suggestion to get (optional)] --path_to_model [path to model checkpoint (optional)]
```

### Train and evaluate
If you want to fit SVD by your self and evaluate the results. Please, viist ```2.0-model-evaluation.ipynb``` to see the examples.