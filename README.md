# Grape Leaf Disease Classification

A CNN in Keras that classifies grape leaf photos as Black Rot, ESCA (Black Measles), Leaf Blight or Healthy.

## Results

- **Test accuracy: 98.5%** (test loss 0.051) on the 1,443-image test folder. The test folder was not used to choose the model.
- A majority-class guess would score about 27% on the training folder (ESCA is 1,536 of 5,779 images).
- Across 10 experiments, validation accuracy ranged from 67% to 97.6%. The notebook selects Experiment 4 (96.9% validation accuracy). See the table below.

| # | Change from the base model | Validation accuracy | Validation loss |
|---|---|---|---|
| 1 | Base model | 97.56% | 0.072 |
| 2 | + BatchNormalization | 89.81% | 0.290 |
| 3 | + Dropout | 71.97% | 0.990 |
| 4 | + extra convolutional layer (**selected**) | 96.90% | 0.140 |
| 5 | Optimizer changed to RMSprop | 86.20% | 0.569 |
| 6 | No augmentation (Exp. 4 model) | 92.30% | 0.217 |
| 7 | Dense layer with 512 nodes (Exp. 4 model) | 92.47% | 0.233 |
| 8 | Batch size 16 (Exp. 4 model) | 78.89% | 0.512 |
| 9 | Dense layer size divided by 2 (Exp. 4 model) | 90.19% | 0.271 |
| 10 | tanh activations (Exp. 4 model) | 67.04% | 0.991 |

## Problem

Grape leaves are hit by several diseases that reduce yield and wine quality. Vineyards are large and sparse, so checking every plant by hand is slow and costly. The notebook sets the goal as a model that takes leaf photos from cameras placed around the plantation and sorts each leaf into one of four classes. Plantation staff then decide whether to remove the leaf or treat the plant.

## Data

- **Source:** [Grape Disease dataset (original)](https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original) on Kaggle. Four classes: Black Rot, ESCA, Healthy, Leaf Blight.
- **Images:** RGB close-up photos of leaves, 256 x 256 pixels in the files the notebook checks. They are resized to 128 x 128 for training.
- **Split used in the notebook:**

| Split | Folder | Images | Black Rot / ESCA / Healthy / Leaf Blight |
|---|---|---|---|
| Train | `Original Data/train` | 5,779 | 1,511 / 1,536 / 1,354 / 1,378 |
| Validation | `Original Data/test` | 1,805 | 472 / 480 / 423 / 430 |
| Test | `Original Data/testing` | 1,443 | not broken down in the notebook |

## Approach

- **Input:** pixel values scaled to 0-1 and images resized to 128 x 128. Colour is kept because the notebook notes that leaf colour differs between classes.
- **Augmentation (training set only):** zoom up to 20%, shifts up to 15% in each direction, brightness between 70% and 140%. These simulate framing and lighting changes outside the lab.
- **Model:** three blocks of Conv2D (3 x 3, ReLU), BatchNormalization and MaxPooling with 32, 64 and 128 filters. Then Flatten, Dense (128, ReLU), Dropout (0.5), BatchNormalization and a 4-way softmax.
- **Training:** Adam (learning rate 0.001), categorical cross-entropy, batch size 32, up to 100 epochs. EarlyStopping (patience 5, on validation accuracy) restores the best weights, and ModelCheckpoint keeps the best validation checkpoint.

Architecture diagrams:

| Base model (Experiment 1) | Experiment 4 model (selected) |
|---|---|
| ![Base model architecture](base_model.png) | ![Experiment 4 model architecture](best_model.png) |

The base model diagram is drawn in the notebook (cell 76). The notebook does not contain the training cell for it. Its metrics come from the experiment table.

## Repo contents

| File | What it is |
|---|---|
| `grape.ipynb` | The full analysis: data checks, augmentation, model, the 10 experiments and test evaluation. Outputs are saved from the original run. |
| `Grape Leaf Classification.html` | HTML export of the notebook with outputs, viewable without Python. |
| `base_model.png` | Diagram of the base model (Experiment 1). |
| `best_model.png` | Diagram of the Experiment 4 model. |

The trained model files are not in the repo. Running the notebook creates them. Cell 61 saves the Experiment 4 model as `base_model.keras`, so that file name does not refer to the base model.

## How to run

This repo contains only the notebook. There are no scripts or tests.

1. Download the dataset from the Kaggle link above. Put it in a folder called `Original Data` next to the notebook, with `train`, `test` and `testing` subfolders. Each subfolder holds one folder per class. The notebook reads all three names, so rename the Kaggle folders to match if needed.
2. Install the packages the notebook imports:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn plotly scikit-learn pillow nbformat nbconvert jupyter
   ```
3. Start Jupyter and run all cells:
   ```bash
   jupyter notebook grape.ipynb
   ```

Notes:
- The saved outputs come from a Windows machine with Python 3.12. The author's training run took about 1,220 seconds (cell 60).
- Cell 45 builds the model inside `tf.device('/GPU:0')`. This was not tested on a machine without a GPU. If you do not have one, remove that `with` block.

## Limitations and next steps

- **Data:** every image comes from one public dataset of close-up leaves. The notebook planned tests on field photos from three plantation regions, but none are in the repo. The 98.5% test score does not show how the model does on field images. Next: add field photos and more disease classes.
- **Single run:** no random seed is set, and the test score comes from one training run. Next: repeat training with fixed seeds to measure the spread.
- **Model choice:** Experiment 1 has higher validation accuracy (97.6% vs 96.9%) and lower validation loss (0.072 vs 0.140) than the selected Experiment 4. The notebook names Experiment 4 as the best of the 10 experiments and does not give a reason for preferring it over Experiment 1. Next: plot learning curves for both models.
- **Per-class results:** the notebook calls precision and recall "moderate" but does not write up per-class scores. Next: add a per-class report and try transfer learning (ResNet, EfficientNet), as the notebook suggests.

Coursework for M507B Method of Prediction.
