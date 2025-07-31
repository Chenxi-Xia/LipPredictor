# Usage Guide​
## Step 1: Environment Setup​
conda create --name LIP python=3.11      # Create Conda environment named LIP with Python 3.11

conda activate LIP                      # Activate LIP environment
## Step 2: Install Dependencies​
pip install -r requirements.txt  # Install required Python packages
## Step 3: Download ProtTrans Model Files​
Download the prot_t5_xl_uniref50 model files from:https://github.com/agemagician/ProtTrans

Place downloaded files in the prot_t5_xl_uniref50 directory.
## ​​Step 4: Generate Embedding Files​
1.Open Embedding_Generation.py.

2.Modify path variables as shown:

dataset_path = "dataset/TR1000.fasta"     # Training set FASTA path

output_path = "embedding/train_embeddings.pkl"  # Output embedding path

3.Objective:​​ Generate corresponding .pkl embedding files in the embedding directory for:

Training set: TR1000.fasta

Test set: TE197.fasta

Or your own dataset.

## Step 5: Train Model​
1.Open Model_Training.py.

2.Configure input paths:

TRAIN_PATH = "dataset/TR1000.fasta"    

TEST440_PATH = "dataset/TE197.txt"                   # Test set file (TE197)

EMBEDDINGS_BASE_PATH = "embedding/{}_embeddings.pkl"  

3.Run the script.

​​Output:​​ Trained model saved in best_model directory (typically best_model.pth).

## Step 6: Generate Test Set Predictions​
Ensure best_model.pth exists in best_model.

1.Open Prediction_Result_Generation.py.

2.Verify these paths are correct:

Pretrained model path

Test data input path

Prediction output path

3.Execute the script.

​​Output:​​ Prediction results saved in prediction directory.
