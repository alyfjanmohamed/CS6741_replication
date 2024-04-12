# CS6741 Replication Project
My project aims to replicate Table 1 from [DRONE: Data-aware Low-rank Compression for Large NLP Models](https://proceedings.neurips.cc/paper_files/paper/2021/file/f56de5ef149cf0aedcc8f4797031e229-Paper.pdf)
![image](https://github.com/alyfjanmohamed/CS6741_replication/assets/51303841/5e7dbfd4-e368-403c-a45f-a79144efcdef)

### Summary of Replication Project
I tried to replicate the previous table with adjustments made to reduce the computational requirements. I believe I was relatively successful showing that the total rank of all matrices can be reduced by approximately 50% without a significant decrease in performance on benchmarks. The two main changes I made to the procedure were: 1) a smaller baseline model (TinyBERT instead of BERT) and 2) simpler rank-reduction rules.

### Details of Replication Project

My version of this table is below. Note, each bechmark is only evaluated once and some benchmarks are missing because I did not have sufficient compute to evaluate them. In addition, CoLA uses evaluation loss as the metric (instead of Matthew's correlation) since the output from the script I was using always gave a value of 0 for this. Also, I have used their methodology on the TinyBERT model instead of BERT to reduce the computation required.

| Model    | RTE | CoLA | STS-B|  WNLI | MRPC |
| -------- | ------- | ------- | ------- | ------- | ------- |
| TinyBERT  |  0.639   |  0.6029   |   0.8383  | 0.5634 | 0.851 |
| Small |   0.6318   |  0.6155   |   0.807  |   0.3521  | 0.809 |
| Medium    |  0.6679   |   0.6052  |  0.8093   |  0.5634   | 0.8439 |
| Int-expanded |  0.6462   |  0.6137   |   0.8113  |   0.4789  | 0.8303 |

In the paper, the original model corresponds to a pre-trained BERT model. SVD corresponds to just applying SVD (and truncating rank) to each weight matrix while DRONE corresponds to the method proposed in the paper (effectively applying SVD to the weight matrix times a sample of inputs). DRONE retrain corresponds to fine-tuning the DRONE model.
#### Rank Selection Algorithm
In the paper, DRONE is applied to both attention mechanism and feed forward networks (attention, intermediate, output) for each of the 12 layers and they propose an algorithm for rank selection. The rank of the approximation increases until an error threshold is reached, this error threshold is based on the inference time of each module. In total, the DRONE rank is approximately 50% of the rank of the BERT model (corresponds to my medium model).
There are 2 key changes in my replication.
1) I only apply DRONE to the feed-forward networks (because I think there might be a typo in their expression for attention)
2) I use much simpler rules for selecting the rank of each layer. The motivation for this is to reduce the compute required. In their methodology, every rank considered requires the model to be re-run to estimate error.

Below is the description of the 3 models in the table. All models are derived from the TinyBERT model [add citation]:
* Small - each feed forward network weights is compressed by 75% to rank 78 (from 312)
* Medium - each feed forward network weights is compressed by 50% to rank 156 (from 312)
* Int-expanded - Attention and output feed forward networks are compressed to rank 78 (from 312) while the intermediate feed forward networks are rank 156

The reason for the int-expanded network was that the largest error between the full rank and reduced rank matrices was always for the intermediate feed forward layer.
![image](https://github.com/alyfjanmohamed/CS6741_replication/assets/51303841/2fefa718-ddc7-4657-8f54-0e2a7e30e9a1)

### Evaluation
The paper identifies that their methodology is successful because their DRONE network has similar performance to the baseline (<3% difference). My medium model (which corresponds to the same level of rank reduction) shows similar results across the benchmarks. Therefore, I believe I was successful replicating their results from this table. In addition, I show that more aggressive rank reduction methods can preserve performance across some (but not all) benchmarks. The benchmark that the smallest models struggle with the most, WNLI, is not reported in the original paper.

### Summary of Code
1) Storing model inputs: To generate sample inputs to each layer, I used the [abokbot/wikipedia-first-paragraph](https://huggingface.co/datasets/abokbot/wikipedia-first-paragraph) dataset. I sample 5000 paragraphs and gave it to the model, storing 5000 randomly selected inputs for each layer. The code can be found in this [notebook](https://github.com/alyfjanmohamed/CS6741_replication/blob/main/tinyBERT_replication_1.ipynb)
2) Reducing rank of matrices: I followed the paper's procedure for finding low-rank representations for the weight matrices given the inputs saved from the previous section. The code can be found in this [notebook](https://github.com/alyfjanmohamed/CS6741_replication/blob/main/tinyBERT_replication_2.ipynb)
3) Evaluation: Finally, for evaluation, I used [run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) from huggingface. The code can be found in this [notebook](https://github.com/alyfjanmohamed/CS6741_replication/blob/main/project_evaluation.ipynb)
I consulted ChatGPT to write some of the code in the notebooks.

### Final Project Plan
Below are 3 paper references, the first one I presented in class
1) [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/pdf/2403.17887.pdf)
2) [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/pdf/2403.03853.pdf)
3) [Importance Estimation for Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf)
4) [Compressing Large Language Models by Streamlining the Unimportant Layer](https://arxiv.org/pdf/2403.19135.pdf)

For my class project, I plan to investigate different methods for select layers to drop from an LLM. The paper I presented claims that almost 50% of layers can be dropped before performance on tasks is affected. I want to investigate 2 ways to improve on this paper: 1) better mechanism for selecting which layers to drop and 2) more thorough investigation of performance on benchmarks. For a better mechanism, I believe that a better metric can be found and that layers do not need to be removed sequentially. I plan to explore how dynamic programming and a greedy heuristic could be used for selecting layers to delete.

An experiment that I plan to run is to take a large open-weights model and iteratively remove layers according to the new mechanism. For each model, I will fine-tune it on a specific task and then measure the performance. I will then plot how the model performs as its layers are removed. A positive result would be showing that performance on a variety of benchmarks does not decay as moderate (30-45%) of layers are removed. A negative result would be the rapid decay of performance as layers are removed.
