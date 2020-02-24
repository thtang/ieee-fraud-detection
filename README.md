# ieee-fraud-detection
## The emperical study on tackling class imbalance using sampling techniques
A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and / or adding more examples from the minority class (over-sampling).

|sampling method   | w/o sampling	| random over-sampling  	|  random under-sampling 	|  SMOTE (Over-sampling) 	|  Tomek Links (under-sampling) 	|  Edited Nearest Neighbours (under-sampling)	|
| :-:	| :-:	| :-:	| :-:	| :-:	| :-:	| :-: |
| Public AUC  	|  0.9251 	|  0.9229 	|  **0.9265** 	| 0.9115  	|  0.9259 	| 0.9248    |
|  Private AUC 	|  **0.9050** 	|  0.8867 	|  0.8976 	| 0.8939  	|  0.9047 	| 0.9022    |
