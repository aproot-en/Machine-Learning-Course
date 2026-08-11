
# ML-01: Fundamentals of Machine Learning (ML)

## 📘 Course Description

Fundamentals of machine learning, supervised and unsupervised learning, classification, clustering, decision tree, k-nearest neighbors, artificial neural networks, support vector machine, genetic algorithm.

## 📊 Assessment Methods

| Component | Points | Description |
|---|---|---|
| Class Participation | 10 | Attend class and join activities and group work |
| Lab Test | 10 | One-on-one practical coding test |
| LAB | 20 | Practice using ML through simple coding tasks |
| Midterm Exam | 30 | Tests understanding of ML ideas and problem-solving with code |
| Final Exam | 30 | Mini project in ML for an application system |


## 🛠️ Software / Tools

- **Python** — Versatile programming language with rich libraries for data science, AI, and ML
- **TensorFlow** — Open-source platform for building and deploying ML models
- **Visual Studio Code** — Powerful, lightweight code editor with rich extensions and integrated tools

---- 

## 🕰️ Historical Context
> "I propose to consider the question, 'Can machines think?'" — **Alan Turing, 1950**
> ([https://doi.org/10.1093/mind/LIX.236.433](https://doi.org/10.1093/mind/LIX.236.433))


### ML Timeline

| Year | Model              | Person / Org                         | Reference                                                                    |
| ---- | ------------------ | ------------------------------------ | ---------------------------------------------------------------------------- |
| 1943 | Neuron Model       | Warren McCulloch & Walter Pitts      | [McCulloch & Pitts, 1943](https://doi.org/10.1007/BF02478259)                |
| 1958 | Perceptron         | Frank Rosenblatt                     | [Rosenblatt, 1958](https://doi.org/10.1037/h0042519)                         |
| 1986 | Backpropagation    | Rumelhart, Hinton & Williams         | [Rumelhart et al., 1986](https://doi.org/10.1038/323533a0)                   |
| 1989 | CNN                | Yann LeCun et al.                    | [LeCun et al., 1989](https://doi.org/10.1162/neco.1989.1.4.541)              |
| 1995 | SVM                | Corinna Cortes & Vladimir Vapnik     | [Cortes & Vapnik, 1995](https://doi.org/10.1007/BF00994018)                  |
| 1997 | LSTM               | Sepp Hochreiter & Jürgen Schmidhuber | [Hochreiter & Schmidhuber, 1997](https://doi.org/10.1162/neco.1997.9.8.1735) |
| 2009 | ImageNet           | Fei-Fei Li et al.                    | [Deng et al., 2009](https://ieeexplore.ieee.org/document/5206848)            |
| 2017 | Transformer        | Google                               | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)                     |
| 2020 | Vision Transformer | Google Research                      | [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)                 |

---

## 🧠 AI, ML, and DL — The Relationship

- **Artificial Intelligence (AI):** The broad field of building intelligent systems that can perform tasks normally requiring human intelligence.
- **Machine Learning (ML):** A subset of AI that enables computers to learn patterns from data and make decisions or predictions without being explicitly programmed.
- **Deep Learning (DL):** A subset of ML that uses multi-layer neural networks to learn complex patterns from large amounts of data.

**Key Takeaway:** AI is the big picture, ML is how machines learn from data, and DL is a powerful subset of ML using deep neural networks.

### Key Areas within ML

| # | Area | Description |
|---|---|---|
| 1 | Supervised Learning (SL) | Learn from labeled data to predict outcomes or continuous values (e.g., classification, regression) |
| 2 | Unsupervised Learning (UL) | Discover hidden patterns or structures in unlabeled data (e.g., clustering, dimensionality reduction) |
| 3 | Reinforcement Learning (RL) | Learn by interacting with an environment and receiving rewards or penalties to maximize cumulative reward |
| 4 | Computer Vision (CV) | ML techniques used to analyze and interpret images or videos |
| 5 | Natural Language Processing (NLP) | ML techniques used to understand, interpret, and generate human language |
| 6 | Time Series & Sequential Models (TS) | Analyze and forecast time-ordered sequential data (e.g., LSTM, Transformers) |

---

## ❓ What is Machine Learning?

> "ML is any process by which a system improves performance from experience."

> "Field of study that gives computers the ability to learn without being explicitly programmed."


### Traditional Programming vs. Machine Learning

| | Input | Process | Output |
|---|---|---|---|
| **Traditional Programming** | Input + Program | Computer | Output |
| **Machine Learning** | Data + Output | Computer | Program (Model) |

---

## ⚖️ Why ML?

**Advantages**
- Learns features from raw data
- Updates with new data (real-time)
- Performs well on unseen/new test data
- Applies to many AI areas (Vision, NLP, RL)

**Limitations**
- Sensitive to bad data
- Needs many data samples
- Takes a long time and needs strong computers

### Three Core Paradigms

| # | Type | Learns From | Output |
|---|---|---|---|
| 01 | **Supervised Learning** | Labeled data | Prediction |
| 02 | **Unsupervised Learning** | Unlabeled data | Clusters / Patterns |
| 03 | **Reinforcement Learning** | Interaction with an environment (reward/penalty) | Optimal action policy |

**In Summary:** ML helps computers learn from data, discover patterns, and make accurate predictions or decisions without being explicitly programmed.

---

## 1️⃣ Supervised Learning (SL)

Supervised Learning (SL) learns a function that maps an input to an output based on labeled input-output examples. — **[Mitchell, T. M. (1997)](https://www.cs.cmu.edu/~tom/mlbook.html)**

### SL: Learning with Labeled Data

| Stage | Description |
| --- | --- |
| **Input Data** | Labeled data containing input features and correct answers |
| **Model Training** | The model learns patterns between inputs and labels |
| **Learning** | The model compares predictions with correct labels and improves |
| **Prediction** | The trained model makes predictions on new data |
| **Output** | Predicted classes or values are produced |

### SL: The Goals of SL

Supervised Learning aims to learn the relationship between input data and known target outputs so that the model can accurately predict outputs for new, unseen data.

**Goals:**

1. **Learn from labeled data** — use examples with known correct answers to learn patterns
2. **Predict classes** — classify new data into predefined categories
3. **Predict values** — estimate continuous numerical values from input features
4. **Minimize prediction error** — reduce the difference between predicted and actual outputs
5. **Generalize to new data** — make accurate predictions on unseen data

### SL: Popular Models

- **Linear Regression** — Predicts continuous values from input features
- **Logistic Regression** — Predicts class labels or probabilities
- **k-Nearest Neighbors (KNN)** — Classifies data based on its nearest neighbors
- **Support Vector Machine (SVM)** — Finds an optimal decision boundary between classes
- **Decision Tree** — Learns decision rules from labeled data
- **Random Forest** — Combines multiple decision trees
- **Neural Network (NN)** — Learns complex patterns using interconnected layers

### SL: Summary

- Uses **labeled data**
- Learns the relationship between **Input → Output**
- Main tasks: **Classification and Regression**
- Learns by minimizing prediction errors
- Predicts outputs for new, unseen data

---

## 2️⃣ Unsupervised Learning (UL)

Unsupervised Learning (UL) learns from unlabeled data without predefined correct answers or target outputs. The model analyzes the data to discover hidden patterns, similarities, relationships, or structures.

### UL: Learning with Unlabeled Data

| Stage | Description |
| --- | --- |
| **Input Data** | Unlabeled data without predefined correct answers |
| **Model Training** | The model analyzes the characteristics of the data |
| **Learning** | The model discovers hidden patterns, similarities, or structures |
| **Processing** | The discovered patterns are used to organize or represent the data |
| **Output** | Clusters, representations, reduced features, or anomalies are produced |

### UL: The Goals of UL

Unsupervised Learning aims to discover useful structures, patterns, and relationships hidden within data without using predefined target labels.

**Goals:**

1. **Discover hidden patterns** — identify structures or patterns in the data
2. **Group similar data** — organize similar data points into clusters
3. **Reduce data complexity** — reduce dimensions while preserving important information
4. **Detect anomalies** — identify unusual data points
5. **Learn data representations** — create meaningful representations of the data

### UL: Popular Models

- **k-Means Clustering** — Divides data into k groups based on similarity
- **DBSCAN** — Groups data based on density and identifies noise
- **Gaussian Mixture Model (GMM)** — Uses multiple Gaussian distributions for probabilistic clustering
- **Autoencoder** — Learns compressed representations and reconstructs the original input
- **Principal Component Analysis (PCA)** — Reduces data dimensions while preserving important information

### UL: Summary

- Uses **unlabeled data**
- No predefined correct answers
- Discovers patterns and structures automatically
- Main tasks: **Clustering, Dimensionality Reduction, and Anomaly Detection**
- Useful for exploring and understanding data

---

## 3️⃣ Reinforcement Learning (RL)

Reinforcement Learning (RL) is a learning approach in which an agent learns by interacting with an environment, taking actions, and receiving rewards or penalties. The goal is to learn a policy that maximizes the total reward over time. — **[Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html)**

### RL: Learning Through Interaction with the Environment

| Stage | Description |
| --- | --- |
| **Environment** | The world or system in which the agent operates |
| **State** | The current situation observed by the agent |
| **Action** | The decision or action selected by the agent |
| **Reward** | Feedback received after performing an action |
| **Learning** | The agent improves its policy based on experience |
| **Output** | A learned policy for selecting appropriate actions |

### RL: The Goals of RL

Reinforcement Learning aims to enable an agent to learn the best actions through interaction with an environment in order to maximize cumulative reward.

**Goals:**

1. **Learn from interaction** — gain experience by interacting with the environment
2. **Maximize cumulative reward** — select actions that produce the highest total reward
3. **Learn an optimal policy** — determine which action should be taken in each state
4. **Balance exploration and exploitation** — explore new actions while using known successful actions
5. **Adapt to the environment** — improve decisions based on experience and feedback

### RL: Popular Models

- **Q-Learning** — Learns the expected reward of actions in different states
- **Deep Q-Network (DQN)** — Combines Q-Learning with deep neural networks
- **Proximal Policy Optimization (PPO)** — Uses controlled policy updates for stable training
- **Deep RL Agent** — Uses deep neural networks to learn complex policies
- **Multi-Agent Reinforcement Learning (MARL)** — Multiple agents learn to cooperate or compete
- **Hierarchical RL Agent** — Divides complex tasks into high-level goals and lower-level actions
- **Model-Based RL Agent** — Uses a model of the environment to predict outcomes and plan actions

### RL: Summary

- Learns through **interaction with an environment**
- Uses **rewards and penalties**
- Learning cycle: **State → Action → Reward → Next State**
- Main goal: **Maximize cumulative reward**
- Common applications: **Robotics, Autonomous Systems, Games, and Intelligent Agents**

---

## 📌 Comparative Summary Table — Types of Machine Learning

| Feature | Supervised Learning | Unsupervised Learning | Reinforcement Learning | Reference |
| --- | --- | --- | --- | --- |
| **Learning Data** | Labeled data | Unlabeled data | Environment interaction | [Bishop, 2006](https://link.springer.com/book/9780387310732) / [Sutton & Barto, 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) |
| **Learning Signal** | Correct labels / target values | No explicit labels | Reward signal | [Bishop, 2006](https://link.springer.com/book/9780387310732) / [Sutton & Barto, 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) |
| **Main Goal** | Predict outputs | Discover patterns or structures | Maximize cumulative reward | [Bishop, 2006](https://link.springer.com/book/9780387310732) / [Sutton & Barto, 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) |
| **Main Tasks** | Classification, Regression | Clustering, Dimensionality Reduction | Decision-making, Control | [Bishop, 2006](https://link.springer.com/book/9780387310732) / [Sutton & Barto, 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) |
| **Output** | Class or value | Cluster or representation | Policy or action strategy | [Bishop, 2006](https://link.springer.com/book/9780387310732) / [Sutton & Barto, 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) |
| **Example** | Cat/Dog classification | Customer clustering | Robot navigation | [Bishop, 2006](https://link.springer.com/book/9780387310732) / [Sutton & Barto, 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) |

---

## 📌 Types of ML by Example

| Example | Type of ML | Input | Output | Reference |
| --- | --- | --- | --- | --- |
| **Spam email filter** | Supervised Learning | Email text | Spam or Not Spam | [Sahami et al., 1998](https://aaai.org/papers/055-ws98-05-009/) |
| **Face recognition** | Supervised Learning | Face image | Person / Identity | [Taigman et al., 2014](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Taigman_DeepFace_Closing_the_2014_CVPR_paper.html) |
| **House price prediction** | Supervised Learning | House size, location | Predicted price | [Harrison & Rubinfeld, 1978](https://doi.org/10.1016/0095-0696%2878%2990006-2) |
| **Fraud detection** | Supervised Learning | Payment history | Normal or Fraud | [Bhattacharyya et al., 2011](https://doi.org/10.1016/j.dss.2010.08.008) |
| **Customer segmentation** | Unsupervised Learning | Customer information | Customer groups | [MacQueen, 1967](https://projecteuclid.org/euclid.bsmsp/1200512992) |
| **Anomaly detection** | Unsupervised Learning | Sensor or transaction data | Normal / Anomaly pattern | [Chandola et al., 2009](https://doi.org/10.1145/1541880.1541882) |
| **Robot navigation** | Reinforcement Learning | Sensor and environment state | Movement action | [Sutton & Barto, 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) |
| **Game AI (Chess, Go)** | Reinforcement Learning | Game state | Action / Move | [Silver et al., 2018](https://doi.org/10.1126/science.aar6404) |
| **Autonomous agent** | Reinforcement Learning | Environment state | Action / Decision | [Sutton & Barto, 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) |

---

---

## ✅ Summary

### Conclusion

- We learned the differences between **Supervised, Unsupervised, and Reinforcement Learning**
- Each learning type uses a different learning signal and solves different types of problems
- Selecting the appropriate learning approach depends on the available data and the problem to be solved

### ML Challenge

- How can we build a lightweight model for mobile or edge devices?
- How can we train a model effectively with limited training data?
- How can an intelligent agent learn and adapt to a changing environment?

### Key Takeaways

- **Supervised Learning:** Labeled Data → **Predict**
- **Unsupervised Learning:** Unlabeled Data → **Discover**
- **Reinforcement Learning:** Environment + Reward → **Decide and Act**

---






















