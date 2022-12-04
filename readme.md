# 3D MRI Brain Tumor Segmentation With Vision Transformer and Modality Fusion

### Getting Started
This is the code for the final project ELEC 6910X.

Install the necessary requirements from requirement.txt

```
conda create --name <env> --file requirements.txt
```

Install Kaggle 2020 dataset at https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation 

Install Kaggle 2021 dataset at https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

Modify the dataset directory on config.py if necessary. The extension of the training data is ".nii", while
the extension for test data is ".nii.gz". 

Modify config.py to change the experiment configurations. 

Data transformation is provided at data_transform.py

Before training manually create "results" folder, "graphs" subfolder, "metrics" subfolder, and "trained_models" subfolder. Also create "Exp" folder for monitoring the training.

To start the training run

```
python main.py

```

To monitor the training (e.g. dice loss) use the following command

```
tensorboard --logdir EXP_PATH

```

where EXP_PATH depends on config.py (default Exp).

To evaluate the model on test dataset and see the samples, modify the config.py EXP_NAME, LOAD_MODEL_NAME (the file name of the pretrained model), and the model configs and run

```
python evaluate.py

```

Samples will be shown on results/samples folder

### Model Architecture



