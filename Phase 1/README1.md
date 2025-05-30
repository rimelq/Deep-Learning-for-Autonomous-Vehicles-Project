# End-to-End Planning for Autonomous Driving · Phase 1

> EPFL — CIVIL-459 “Deep Learning for Autonomous Vehicles”  
> Milestone 1 — **deadline 2 May 2025**

> **Team** : Rim El Qabli *(Sciper 340997)*, Mathéo Taillandier *(Sciper 339442)*

---

## 1. Repository layout

| Path                      | Purpose                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------------- |
| DLAV_Phase1_Final.ipynb   | A self-contained notebook that trains, validates, visualizes, and exports the Phase 1 planner. |
| README.md                 | You are reading it. Describes structure and design choices.                                   |
| submission_phase1.csv     | Output CSV file formatted for submission used on the [Kaggle DLAV_phase_1 competition](https://www.kaggle.com/t/338eec1b2cd346eaa3b569340ab2de19)..                                                    |
| images/loss_vs_epoch.png  | Visualization of training and validation losses over epochs.                                                  |
| images/sample_predictions.png | Example predictions visualized from the model output. 

Everything needed to run our code is inside the notebook.

---

## 2. Task recap

Given the inputs:

- An RGB dash-cam frame at time step **t**  
- The vehicle’s past **21** ego-centric poses *(x, y, heading)*  

Predict the next **60** poses.

**Allowed inputs:** camera image · driving command *(left / right / forward)* · past trajectory  
**Metric:** **Average Displacement Error (ADE)** — lower is better. The goal in this milestone is to obtain an **ADE score less than 2** by uploading the generated `submission_phase1.csv` to the [Kaggle DLAV_phase_1 competition](https://www.kaggle.com/t/338eec1b2cd346eaa3b569340ab2de19).


---

## 3. Environment

We tested on:

| Setup | GPU               | Approx. Training Time |
| ----- | ----------------- | --------------------- |
| Colab | NVIDIA T4 or A100 | ~45 min               |


---

## 4. Starter code base architecture Analysis

=> "*By running the code above, you have trained your first model! What do you observe in terms of train and val losses? What is this phenomenon called?* "

The training loss shows a consistent decline, dropping from very high initial values to much lower ones by the end of training. In contrast, the validation loss decreases initially but then begins to plateau and fluctuate, remaining noticeably higher than the training loss. This pattern indicates that the model is fitting the training data too closely, capturing specific details and noise rather than learning generalizable features. As a result, its performance on new, unseen data is worse. This behavior is known as overfitting, where a model becomes too specialized to the training set and fails to generalize effectively to other data.

---

## 5. Our architecture implementation

We designed a lightweight but effective planner with four key components:

1. **Image encoder (CNN vs. ViT)**  
   - A five-stage convolutional network processes 200×300 RGB frames.  
   - Chosen over ViTs because our ~5000-image dataset leads transformers to overfit without extensive data augmentation or pretraining.

2. **Trajectory encoder (LSTM)**  
   - A single-layer LSTM (hidden = 256) ingests the past 21 poses, capturing temporal dependencies in position and heading.  
   - Outperformed GRUs and temporal convolutional layers, particularly in maintaining consistent heading predictions.

3. **Late fusion**  
   - We concatenate the 512-dim CNN embedding with the 256-dim LSTM hidden state immediately before the prediction head.  
   - Early- or mid-level fusion increased compute by >2× for negligible accuracy gains, so we kept it simple and efficient.

4. **Residual prediction head**  
   - A 4-layer MLP with a linear shortcut: the network predicts a coarse offset which is added to a direct linear projection of the fused embedding.  
   - This residual connection stabilizes gradients in early epochs and reduced ADE.

---

## 6. Training hyperparameters

| Hyperparameter       | Value                                                         |
| -------------------- | ------------------------------------------------------------- |
| Batch size           | 32                                                            |
| Optimizer            | AdamW                                                         |
| Initial LR           | 3 × 10⁻⁴                                                     |
| Weight decay         | 1 × 10⁻³                                                     |
| Scheduler            | ReduceLROnPlateau (patience 5, factor 0.5)                    |
| Epochs               | 50                                                           |
| Data augmentation    | 50 % horizontal flip (image and sign flip of x-coordinates)   |

---

## 7. Results

| Split               | ADE ↓      | Notes                                         |
| ------------------- | ---------- | --------------------------------------------- |
| Kaggle public test  | **1.66331 m**  | Team on the Kaggle leaderboard : ***Mathéo & Rim*** |

We obtained the following output visualizations from our model:

![Sample predictions](images/sample_predictions.png)
![Loss curves](images/loss_vs_epoch.png)

The training and validation loss curves (see graph above) show:

- **Rapid initial convergence**: Both train and val losses drop sharply in the first 5–10 epochs, meaning the model quickly learns the core scene-to-trajectory mapping.  
- **Slower later improvement**: After epoch 10, train loss continues its downward trend while validation loss flattens around 0.55–0.60, suggesting that additional epochs yield only marginal gains on unseen data.
From ~epoch 35 onward, the gap widens (train < 0.40 vs. val ≈ 0.55), showing the model fits the training set a bit more tightly but still maintains very reasonable generalization.  
- **Consistent final performance**: Despite minor val-loss fluctuations, the public-test ADE of 1.66 m confirms we meet the sub-2 m milestone with room to spare.

---





