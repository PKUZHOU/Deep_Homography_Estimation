# Deep_Homography_Estimation

## Usage:
Step1: Setup the environment:
    
    pip3  -r requirements.txt

Step2: Generate training data and validation data. You need to modify 

    python3 gen_datasets.py --phase train --train_coco_path xxx 
    python3 gen_datasets.py --phase val --val_coco_path xxx

Step3: Train the model

    python3 train.py

Step4: Test the trained model

    python3 test.py --model checkpoints/xxx.pkl