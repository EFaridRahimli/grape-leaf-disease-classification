# Grape Leaf Disease Classification

A CNN that classifies grape-leaf photos as **Black Rot, ESCA (Black Measles), Leaf Blight or Healthy**, so vineyard cameras can flag disease early instead of relying on slow, error-prone manual inspection.

## Results
- **98.5% test accuracy** (test loss 0.051).
- Best architecture: **Experiment 4** (extra convolutional layer), chosen from 10 controlled experiments.

| Baseline model | Best model (Exp. 4) |
|---|---|
| ![Base model](base_model.png) | ![Best model](best_model.png) |

## Approach
1. **EDA** of the [Grape Disease dataset](https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original).
2. **Augmentation** with `ImageDataGenerator` to simulate real field conditions.
3. **Custom CNN:** stacked convolution blocks (32 → 64 → 128 filters), batch normalisation, dropout and a softmax head.
4. **10 experiments** varying depth, batch norm, dropout, optimiser (Adam vs RMSprop), dense-layer size, batch size and augmentation; the best checkpoint is kept by validation accuracy.

## Business value
- Lower labour cost: no manual leaf-by-leaf inspection.
- Faster response: fixed cameras report continuously.
- Protects yield and wine quality.

## Next steps
Broader data (regions, lighting, seasons), more disease classes, stronger augmentation, and transfer learning (ResNet, EfficientNet).

## Tech stack
Python · TensorFlow/Keras · NumPy · pandas · Matplotlib · Seaborn · Plotly · scikit-learn

## Run it
Download the dataset from the Kaggle link above, then:
```bash
pip install tensorflow numpy pandas matplotlib seaborn plotly scikit-learn jupyter
jupyter notebook grape.ipynb
```
`Grape Leaf Classification.html` is a rendered copy of the notebook you can open in a browser.
