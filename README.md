# RLHF Experiments

## About
This application contains my experiments with RLHF.

## Tools
- SentenceTransformers
- HuggingFace datasets
- MLFlow
- PyTorch
- Tqdm

## Sections

### 1. Reward Models (`/reward-models/RLHF_101.ipynb`)
#### 1.1. Dataset
The dataset used here is the [`RLHFlow/Argilla-Math-DPO-standard`](https://huggingface.co/datasets/RLHFlow/Argilla-Math-DPO-standard) dataset from huggingface. Each datapoint contains:
- an input prompt and a human chosen AI response,
- an input prompt and a human rejected AI response, 
- a rating of the human chosen AI response,
- a rating of the human rejected AI response.

#### 1.2. Modelling
In my experiments, I tried different training methods and architecture, some of them were more promising than others. I'll give a TL;DR of each of the approaches here.

##### 1.2.1. Simple NN - Regression Model
This trains a Neural Network to predict the ratings of the ratings of each of the AI responses. So, each AI response, both chosen and rejected is taken as a separate datapoint. There's a pretrained embedding model in the flow, this pretrained embedding model is gotten from huggingface, and used as is. The only model being trained here is our Neural Net. The flow is something like this:

(Human Input Prompt + Chosen AI Response) -> Pretrained Embedding Model -> Neural Net -> Predicted Rating of Chosen Response
(Human Input Prompt + Rejected AI Response) -> Pretrained Embedding Model -> Neural Net -> Predicted Rating of Rejected Response

##### 1.2.2. Simple NN - Classification Model
This trains a Neural Network to predict which output will have the higher rating given two outputs. There's a pretrained embedding model in the flow, this pretrained embedding model is gotten from huggingface, and used as is. The only model being trained here is our Neural Net. So, the flow is like this:

((Human Input Prompt + AI Response 1) + (Human Input Prompt + AI Response 1 2)) -> Pretrained Embedding Model -> Neural Net -> Predict Which Response a Human Would Prefer (Response 1 or 2)

##### 1.2.3. Reward Model - Simple Frozen-Embedding Input
This approach is more sophisticated, it uses a better embedding model to capture the semantics of language, this is what makes this approach superior to the first two approaches. Unlike the previous first and second attempts above that used Mean Squared Error loss and Binary Cross Entropy loss, this attempt uses the Bradley-Terry loss function. Also, as for the embedding model here, this uses the hidden state of a good embedding model, which captures semantics better, but we don't adjust the weights of this embedding model.

(Human Input Prompt + Chosen AI Response) -> Pretrained Embedding Model using Hidden States -> Neural Net -> Predicted Rating of Chosen Response
(Human Input Prompt + Rejected AI Response) -> Pretrained Embedding Model using Hidden States -> Neural Net -> Predicted Rating of Rejected Response

##### 1.2.4. Reward Model - Simple Non-Frozen-Embedding Input (Best Approach)
This approach is way more sophisticated, it uses a good embedding model to capture the semantics of language. Unlike the previous three attempts above, here we also adjust the weights of the embedding model that we got from huggingface. This is the main reason this approach worked best for training. Adjusting the weights of the pretrained model while also training our neural net means that the model learns to adjust it's own embeddings that understands language and also adjusts the reward head that predicts what a good output is. This attempt also uses the Bradley-Terry loss function.

(Human Input Prompt + Chosen AI Response) -> Pretrained Embedding Model that Will be Trained -> Neural Net -> Predicted Rating of Chosen Response
(Human Input Prompt + Rejected AI Response) -> Pretrained Embedding Model that Will be Trained -> Neural Net -> Predicted Rating of Rejected Response

#### 1.3. Limitations
I did these experiments on my personal laptop, and because of the high computational resource and the amount of time it will take to test every tweak, I decided to end the experimentation as soon as I could (I have a day job). As a result, you'll notices there's some overfitting in the training, I intentionally left it like this. This experiment is aimed at showing how to just train a simple RLHF model. I didn't have the time to continue this experiment further.