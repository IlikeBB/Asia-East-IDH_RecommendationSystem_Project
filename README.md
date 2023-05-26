# Asia East - Intradialytic Hypotension Recommendation System
### Stage 1. 
`Construct a system for classifying records in free-text that contain symptoms of Intradialytic Hypotension (IDH).`
<img width="1000" alt="image" src="https://github.com/IlikeBB/IDH_RecommendationSystem_Project/assets/32098079/16a9f059-fa66-45dd-8570-5fac68e5693a">

### Stage 1-1. 
`Data cleaning and enhancement of information in the free-text data.`

### Stage 1-2. 
`Find a suitable NLP Encoder model and use the extracted features for an MLP model.`

| Model  | source url | Ouput embedding |
| ------ | :----------: | :---------------: |
| mUSE   | [Paper](https://arxiv.org/pdf/1907.04307)           | (,512)          |
| mSBERT | [Paper](https://arxiv.org/abs/2004.09813)       | (,512)          |

### Stage 1-3. 
`Determine whether Intradialytic Hypotension occurred during each dialysis session based on the classification results and blood pressure status.`
### Stage 1. Performance 
`Stage 1. Training & Validation Performance`</br>
<img width="600" alt="image" src="https://github.com/IlikeBB/IDH_RecommendationSystem_Project/assets/32098079/5abb54d8-b6d6-44f0-acb6-9203802b2a92">
<img width="600" alt="image" src="https://github.com/IlikeBB/IDH_RecommendationSystem_Project/assets/32098079/8d59df10-2652-4df1-989b-04f6b9bb6f95">
<img width="600" alt="image" src="https://github.com/IlikeBB/IDH_RecommendationSystem_Project/assets/32098079/135f0d5a-ff6f-40c3-8c9a-6aeee229e34a">

`Stage 1. Test Performance`</br>
<img width="600" alt="image" src="https://github.com/IlikeBB/IDH_RecommendationSystem_Project/assets/32098079/46958a77-e8eb-4873-be66-dab57aeafa75">
<img width="600" height ="200" alt="image" src="https://github.com/IlikeBB/IDH_RecommendationSystem_Project/assets/32098079/ba44c930-4b86-4b01-8a4f-5277950db4bc">
***

#### Stage 2. ```Build a recommendation system for the dehydration dosage during each dialysis session.```
#### Stage 2-1. ```Evaluate the total dehydration dosage for each session based on the "weight" and "dry weight" information in the nursing records.```
#### Stage 2-2. ```Identify free-text records with dehydration information and dehydration numerical data in the nursing records using the IDH classification model.```
#### Stage 2-3. ```Provide recommendations for the dehydration dosage for each dialysis session based on past records.```
