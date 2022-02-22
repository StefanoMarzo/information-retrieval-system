# InformationRetrievalSystem

## Installation 
<p>Note: PIP is needed</p> 
1. Create an empty virtual environment 
2. Activate the virtual environment 
3. Move into the project path 
4. Run the following command 

`pip3 install -r requirements.txt --upgrade`

## Setup 
Make sure to have topics in ./topics/ 
Make sure to have collection in ./COLLECTION/ 
Folder 'model_structures' is not necessary 
Note: computing the structures takes long time 

## Run the program 

`python "IR Multiple Language Model.py"` 

## In case of errors 
run the following 

```
pip install numpy
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en-core-web-sm
```

## Output
You can find the OUTPUTS in ./IR_output/ 
Output is divided in sub folders for each model 
We can run trec_eval directly from an output file 

### Thanks 
