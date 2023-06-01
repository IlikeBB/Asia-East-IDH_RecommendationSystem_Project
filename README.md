# Asia East - Intradialytic Hypotension Recommendation System
### Stage 1. 
`Construct a system for classifying records in free-text that contain symptoms of Intradialytic Hypotension (IDH).`
<img width="1000" alt="image" src="https://github.com/IlikeBB/Far-Eastern-Memorial-Hospital-IDH_RecommendationSystem_Project/assets/32098079/f1340c9c-3275-4be2-a5d8-037545f4edde">

### Stage 1-1. 
`Data cleaning and enhancement of information in the free-text data.`

### Stage 1-2. 
`Find a suitable NLP Encoder model and use the extracted features for an MLP model.`

| Model  | source url | Ouput embedding | API |
| ------ | :----------: | :---------------: | :------: |
| mUSE   | [Paper](https://arxiv.org/pdf/1907.04307)           | (,512)          | [API](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3) |
| mSBERT | [Paper](https://arxiv.org/pdf/2004.09813.pdf)       | (,512)          | [API](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1) |

### Stage 1-3. 
`Determine whether Intradialytic Hypotension occurred during each dialysis session based on the classification results and blood pressure status.`
### Stage 1. Performance 
`Stage 1. Training & Validation Performance`</br>
<img width="600" alt="image" src="https://github.com/IlikeBB/Asia-East-IDH_RecommendationSystem_Project/assets/32098079/bafd8e67-cf7d-4324-aa2c-7f6bc210a5d7">
<img width="600" alt="image" src="https://github.com/IlikeBB/Asia-East-IDH_RecommendationSystem_Project/assets/32098079/91436877-2f0f-488c-ac79-fdb7971355e5">
<img width="600" alt="image" src="https://github.com/IlikeBB/Asia-East-IDH_RecommendationSystem_Project/assets/32098079/99a222e4-99e2-4afa-a806-6564babe5382">

`Stage 1. Test Performance`</br>
<img width="600" height ="300" alt="image" src="https://github.com/IlikeBB/Asia-East-IDH_RecommendationSystem_Project/assets/32098079/3c7a264b-619d-427d-9b8c-d87abbad26eb">
<img width="439" alt="image" src="https://github.com/IlikeBB/Far-Eastern-Memorial-Hospital-IDH_RecommendationSystem_Project/assets/32098079/02c8a9a5-c8c3-467a-8254-a154d85ad980">

***

#### Stage 2. ```Build a recommendation system for the dehydration dosage during each dialysis session.```
#### Stage 2-1. ```Evaluate the total dehydration dosage for each session based on the "weight" and "dry weight" information in the nursing records.```
#### Stage 2-2. ```Identify free-text records with dehydration information and dehydration numerical data in the nursing records using the IDH classification model.```
#### Stage 2-3. ```Provide recommendations for the dehydration dosage for each dialysis session based on past records.```
