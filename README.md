# InformationRetrievalSystem

## Installation 
Note: PIP is needed  
Create an empty virtual environment  
Activate the virtual environment  
Move into the project path  
Run the following command  

`pip3 install -r requirements.txt --upgrade`

## Setup 
Make sure to have topics in `./topics/`  
Make sure to have collection in `./COLLECTION/`  
Folder `model_structures` is not necessary, but computing the structures takes long time (up to 2000s)

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

## Understanding the program
The best way to understand the program is to check the NoteBook `IR Multiple Language Model.ipynb`  
it contains the same program and produces the same output, but it also presents explanations.

## Output
You can find the OUTPUTS in `./IR_output/`  
Output is divided in sub folders for each model  
We can run trec_eval directly from an output file  

## Project structure
```
COLLECTION/
  all xml documents here
  
topics/
  all xml queries here
  
model_structures/
  all structures files here
  
IR_output/
  VSM/
    VSM.out
    
  BM25/
    BM25.out
    
  BM25F/
    BM25F.out
    
  ULM/
    ULM.out
    
IR Multiple Language Model.py
IR Multiple Language Model.ipynb
stopwords.txt
requirements.txt
```

### Thanks 
