# CO2Vec: Embeddings of Co-Ordered Networks Based on Mutual Reinforcement
Official implementation of our paper "CO2Vec: Embeddings of Co-Ordered Networks Based on Mutual Reinforcement" (DSAA 2020). CO2Vec is an order representation learning model for co-ordered netwroks. 

## Dependencies
The core learning model is built using [PyTorch](https://pytorch.org/)
* Python 3.6.3
* PyTorch 0.3.0


## Run
To reproduce the results on [UNIV](https://github.com/harrylclc/concept-prerequisite-papers) dataset, the hyperparameters are set in example.sh.

```
  bash example.sh
```

## Data
There should be five data files ready in the 'datasets' folder, e.g. datasets/name/
* ```<name>_split_train.pkl``` list of training instance in pickle format, each instance is a three tuple for type-A entities: (ent_i, ent_j, label), label is either -1 or 1
* ```<name>_split_train_e2.pkl``` list of training instance in pickle format, each instance is a three tuple for type-B entities: (ent_i, ent_j, label), label is either -1 or 1
* ```<name>_split_train_pos.cross.pkl``` list of training instance in pickle format, each instance is a four tuple for cross-entity relations from type-A to type-B entities: (ent_i, ent_j, weight)
* ```<name>_split_train_pos.double.pkl``` list of training instance in pickle format, each instance is a four tuple for cross-entity relations from type-B to type-A entities: (ent_i, ent_j, weight)


## Cite
Please consider cite our paper if you find the paper and the code useful.

```
@inproceedings{CO2Vec2020,
 author = {Meng-Fen Chiang and
            Ee-Peng Lim and 
            Wang-Chien Lee and                
            Philips Kokoh Prasetyo},
 title = {CO2Vec: Embeddings of Co-Ordered Networks Based on Mutual Reinforcement},
 booktitle = {IEEE International Conference on Data Science and Advanced Analytics (DSAA)},
 year = {2020}
} 
```

Feel free to send email to ankechiang@gmail.com if you have any questions. This code is modified from [ANR](https://github.com/almightyGOSU/ANR).
