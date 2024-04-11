# CS6741 Replication Project
Trying to replicate Table 1 from DRONE: Data-aware Low-rank Compression for Large NLP Models (below)
![image](https://github.com/alyfjanmohamed/CS6741_replication/assets/51303841/5e7dbfd4-e368-403c-a45f-a79144efcdef)

My version of this table is below. Note, each bechmark is only evaluated once and some benchmarks are missing because I did not have sufficient compute to evaluate them. In addition, CoLA uses evaluation loss as the metric (instead of Matthew's correlation) since the output from the script I was using always gave a value of 0 for this.

| Model    | RTE | CoLA | STS-B|  WNLI |
| -------- | ------- | ------- | ------- | ------- |
| TinyBERT  |  0.639   |  0.6029   |   0.8383  | 0.5634 |
| Small |   0.6318   |  0.6155   |   0.807  |   0.3521  |
| Medium    |  0.6679   |   0.6052  |     |  0.5634   |
| Int-expanded |  0.6462   |  0.6137   |     |   0.4789  |

In the paper, the original model corresponds to ----. SVD corresponds to just applying SVD (and truncating rank) to each weight matrix while DRONE corresponds to the method proposed in the paper (effectively applying SVD to the weight matrix times a sample of inputs). DRONE retrain corresponds to fine-tuning the DRONE model.

What is Original model?

Below is the description of the 3 models in the table. All models are derived from the TinyBERT model [add citation]:
* Small - each feed forward network weights is compressed to rank 78 (from 312)
* Medium - each feed forward network weights is compressed to rank 156 (from 312)
* Int-expanded - Attention and output feed forward networks are compressed to rank 78 (from 312) while the intermediate feed forward networks are rank 156

The reason for the int-expanded network was that the largest error between the full rank and reduced rank matrices was always for the intermediate feed forward layer.

Explain procedure (+ deviations from table)

# Final Project Plan
Paper references (identify paper that was presented)
1 Paragraph description
1 experiment with action for positive / negative results
