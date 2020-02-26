# IEEE-fraud-detection
## An emperical study on tackling class imbalance using sampling techniques
The dataset for fraud detection is imbalanced with 569,877 positive samples and 20,663 negative samples.
A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and / or adding more examples from the minority class (over-sampling).
<img src="https://github.com/thtang/ieee-fraud-detection/blob/master/figures/resampling.png">

### Methodology:
* Over sampling
  * Random oversampling
  * SMOTE
* Under sampling
  * Random undersampling
  * Tmek Links
  * Edited Nearest Neighbours
  
### Sampling results:
|  sampling method 	| w/o sampling 	| random over-sampling 	| random under-sampling 	| SMOTE (Over-sampling) 	| Tomek Links (under-sampling) 	|
|:----------------:	|:------------:	|:--------------------:	|:---------------------:	|:---------------------:	|:----------------------------:	|
| positive samples 	| 569877       	| 569877               	|         20663         	|         569877        	|            **562770**           	|
| negative samples 	| 20663        	| 569877               	|         20663         	|         569877        	|             20663            	|

It seems that the dataset exist a few of Tomek’s links thus most of the positive samples remained.

###  Experimental results:
|sampling method   | w/o sampling	| random over-sampling  	|  random under-sampling 	|  SMOTE (Over-sampling) 	|  Tomek Links (under-sampling) 	|  Edited Nearest Neighbours (under-sampling)	|
| :-:	| :-:	| :-:	| :-:	| :-:	| :-:	| :-: |
| Public AUC  	|  0.9251 	|  0.9229 	|  **0.9265** 	| 0.9115  	|  0.9259 	| 0.9248    |
|  Private AUC 	|  **0.9050** 	|  0.8867 	|  0.8976 	| 0.8939  	|  0.9047 	| 0.9022    |

According to the experimental results, over-sampling damages the performance and under-sampling methods provide comparable results with much less training time. In short, these kind of tricks make a limited impact on fraud dataset, which is concluded at variance with toy examples discussed in [Tackling Class imbalance](https://www.kaggle.com/shahules/tackling-class-imbalance).

### Reference
[1] https://www.kaggle.com/shahules/tackling-class-imbalance<br>
[2] https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
