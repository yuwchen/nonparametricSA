# Non-parametric Speech Assessment

This repo is the implementation of the paper

*Adaptable non-parametric Approach for Speech-based Symptom Assessment: Isolating Private Medical Data in a Retrieval Datastore*

 <img src="https://github.com/yuwchen/nonparametricSA/blob/main/plot/overview.png" alt="main"  width=100% height=100% />


### Environment
CUDA Version 12.1
Python 3.9
- faiss  1.8.0
- torch                    2.5.0
- torchaudio               2.5.0
- transformers             4.46.2
- scikit-learn             1.5.2
- pandas                   2.2.3

----------
### Data preparing

Organize the train/test list in a .txt file using the following format:
```
/path/to/wavfile.wav;label;Age:{};Sex:{};
```
Please refer to the files in the txtfile directory for examples.

#### Cowsara
- The train/validation/test splits are organized in the txtfile directory.
- [[Download]](https://github.com/iiscleap/Coswara-Data?tab=readme-ov-file) wavfiles. 

#### Covid-19 sounds
- [[Download]](https://www.covid-19-sounds.org/en/) Covid-19 Sounds dataset. Data used in this study is "0426_EN_used_task1". 

-----------
### Datastore building
```
python build_dataset.py --rootdir /path/to/wavfile/dir/ --traintxt /path/to/train_list.txt
```
Datastore will store in a directory named "faiss_database"

------------
### Run assessment
```
python health_assessment.py --rootdir /path/to/wavfile/ --testtxt /path/to/test_list.txt
```

### Limitation
- The results may differ slightly from those reported in the paper, likely due to KMeans randomness and floating-point precision.
- The datastore from one dataset cannot generalize to another because the specific sentences spoken differ across the two and both sentences are short.

