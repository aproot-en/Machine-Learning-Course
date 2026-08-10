
# ML-01: Fundamentals of Machine Learning (ML)

## Course Description

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

- **Python** — Versatile programming language with rich libraries for data science, AI, and machine learning
- **TensorFlow** — Open-source platform for building and deploying machine learning models
- **Visual Studio Code** — Powerful, lightweight code editor with rich extensions and integrated tools
- 

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

> "Machine Learning is any process by which a system improves performance from experience." — **[Herbert Simon](https://en.wikipedia.org/wiki/Herbert_A._Simon)**

> "Field of study that gives computers the ability to learn without being explicitly programmed." — **[Arthur Samuel (1959)](https://doi.org/10.1147/rd.33.0210)**


### Traditional Programming vs. Machine Learning

| | Input | Process | Output |
|---|---|---|---|
| **Traditional Programming** | Input + Program | Computer | Output |
| **Machine Learning** | Data + Output | Computer | Program (Model) |

---

## ⚖️ Why Machine Learning?

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

**In Summary:** Machine Learning helps computers learn from data, discover patterns, and make accurate predictions or decisions without being explicitly programmed.

---

## 1️⃣ Supervised Learning (SL)

SL is learning a function that maps an input to an output based on example input-output pairs. — **[Mitchell, T. M. (1997)](https://www.cs.cmu.edu/~tom/mlbook.html)**  

### SL: Learning with Labeled Data

The model learns from labeled examples to make predictions on new data.

| Stage | Description |
|---|---|
| **Labeled Data** | Examples come with correct answers (labels) |
| **Model Training** | The model learns patterns from the labeled data |
| **Prediction** | The trained model makes predictions on new data |
| **Output** | Predicted labels (e.g., Dog or Cat) are produced |

**Why is it called "Supervised" Learning (SL)?**
1. A "teacher" (Label) provides the correct answer — every training example has a known Spam/Not Spam label
2. It learns the relationship between Input and Label — the model learns patterns from correctly-labeled examples
3. It measures error and improves — Loss + Backpropagation are used to make predictions as close to the true label as possible
4. It predicts new data more accurately — once trained well, the model can accurately predict on new, unseen data

**Summary:** Supervised Learning means having the model "learn from data that already has answers (labels)" in order to build a function that maps Input → Output, and use it to accurately predict the answers for new data.

### SL: Summary & Key Takeaways

- Supervised learning needs labeled data
- Two main tasks: classification & regression
- Common models: Linear/Logistic Regression, Decision Tree, Neural Network
- Key process: Train → Validate → Test

> **Note:** "In SL, experience guides the model — every answer teaches the system to predict better."

---

## 2️⃣ Unsupervised Learning (UL)

UL is learning without a teacher. It means that the model learns from unlabeled data, where no predefined correct answers or target outputs are provided. Instead of being told what to predict, the model analyzes the data by itself to discover hidden patterns, similarities, relationships, or structures. Common applications of unsupervised learning include clustering, dimensionality reduction, and pattern discovery.

### UL: Learning Hidden Patterns without Labeled Data

| Stage | Description |
|---|---|
| **Input Raw Data** | Collection of raw, unlabeled data (e.g., images) |
| **Interpretation** | The model analyzes the data to understand its structure |
| **Model Training** | The model learns hidden patterns without using any labels |
| **Processing** | The learned patterns are used to process and generate output |
| **Output** | The model produces meaningful results (e.g., grouping or class) |

### UL: The Goals of UL

UL is the process of discovering the structure, patterns, and relationships hidden in data that has not yet been labeled — without using any predefined answers.

**Five Key Goals:**
1. **Discover hidden patterns** — find groups, structures, or patterns that are not obvious in the data
2. **Reduce data complexity** — reduce the number of features (variables) or dimensions of the data while preserving key information
3. **Group similar data** — cluster or split data into groups based on natural similarity
4. **Detect anomalies** — identify data points that clearly differ from the majority of the data
5. **Learn data representations** — learn meaningful representations (vectors or representative values) of the data that can reflect key factors or structures within it

**Example Tasks:**
- Clustering (K-Means, Hierarchical Clustering, DBSCAN)
- Dimensionality Reduction (PCA, t-SNE, UMAP, Autoencoder)
- Anomaly Detection (Isolation Forest, One-Class SVM)
- Association Rule Mining (Market Basket Analysis)
- Density Estimation (Gaussian Mixture Models, KDE)

**Key Summary:** UL helps us understand unlabeled data by discovering hidden patterns, reducing complexity, and uncovering meaningful structures or relationships.

### UL: Clustering

Clustering groups similar data points together without using any labels.

- **Raw Data (Unlabeled):** e.g., dog and cat images with no labels
- **Clustering Result:** Data points close to each other are grouped together, each color represents a different cluster

**Definition:** Clustering is grouping data points that look alike into the same group.

**Key Takeaway:**
- Clustering is a type of UL
- It finds natural groupings in data without using labels
- It helps us understand the structure of data

### How DL Works with UL

**Core Idea:** The model learns the structure, patterns, and relationships within data that has no labels (No Label), in order to discover hidden patterns or group the data (Grouping).

**Popular UL Models:**
- **Autoencoder** — Learns to compress data into a low-dimensional latent space, then reconstructs it back to the original form; used for finding key features or detecting anomalies
- **k-Means Clustering** — Divides data into k groups using cluster centroids
- **DBSCAN** — Groups data based on density; can find clusters of irregular shapes and detect noise/outlier points
- **Gaussian Mixture Model (GMM)** — Views data as a mixture of several Gaussian distributions; used for clustering data with complex distributions

### UL: Summary

- No labeled data → the model finds hidden patterns by itself
- Main tasks: clustering, reducing data size, finding relationships
- Used to explore and understand data

---

## 3️⃣ Reinforcement Learning (RL)

> "RL is a way for an agent to learn by trying different actions and learning from rewards. The agent is not told the correct action. Instead, it learns which actions are best by trial and error to get the highest total reward." — **Sutton & Barto, 2018**
> (https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

### RL: Learning Through Interaction with the Environment

An agent learns through trial and error, aiming to obtain the highest cumulative reward over the long term.

**Key Components:**
| Component | Meaning |
|---|---|
| Agent | The learner or decision-maker |
| Action | The action the agent chooses to take (e.g., up, down, left, right) |
| State | The current state observed by the agent |
| Reward | The feedback from the environment (e.g., +1 for reaching the goal, -1 for hitting a wall) |
| Policy | The method or rule the agent uses to make decisions |

**RL Cycle:**
1. The agent observes the current state sₜ
2. The agent selects an action aₜ
3. The environment returns a reward rₜ₊₁ and a new state sₜ₊₁
4. The agent updates its own policy
5. Repeat until the goal is reached or the best policy is found

**Goal:** Learn the best policy in order to obtain the maximum total reward throughout all interactions with the environment.

### How DL Works with RL

**Core Idea:** Learn through trial and error, with the Agent interacting with the Environment to maximize the Cumulative Reward.

| Step | Description |
|---|---|
| 1. Define RL Problem | Identify the goal and components of the RL problem: State, Action, Reward, Policy, Goal |
| 2. Observe State | The Agent receives the current state information from the Environment |
| 3. Select Action | The Agent selects an action aₜ according to its policy π |
| 4. Environment Response | The Environment changes state and provides a reward rₜ₊₁ |
| 5. Learn & Update | The Agent uses its experience (sₜ, aₜ, rₜ₊₁, sₜ₊₁) to update its policy/value estimates |
| 6. Episode Loop | Repeat over many steps until the episode ends, then start a new episode |
| 7. Optimal Policy | Obtain the policy π* that maximizes the cumulative return, ready for real-world deployment |

**Example Applications:** Games (Atari, Go, Chess, StarCraft), robotics and control, recommendation systems (product recommendations, online ads), resource allocation (Wi-Fi bandwidth, energy, cloud resources), finance (algorithmic trading, portfolio management)

**Advantages of RL:**
- Doesn't require labeled data
- Learns directly from experience
- Well-suited for complex and changing problems
- Can discover the optimal strategy

**Limitations of RL:**
- Requires a long training time
- Experimentation/trial can be costly
- Designing an appropriate reward function is difficult
- Training instability issues

**Summary:** Reinforcement Learning means having the Agent learn by interacting with the Environment — trying many actions, receiving rewards as feedback signals, and adjusting its policy to maximize cumulative reward.
 
### RL: Overview

Learn by taking action, receiving rewards, and gradually improving.

**What is RL?** RL learns from actions and outcomes, using reward as the signal of good or bad — there's no fixed correct answer; the agent must try actions and learn from the results.

**How does RL work?**
1. The agent observes the current state from the environment
2. The agent chooses an action
3. The environment responds with a new state and gives a reward
4. The agent learns and adjusts its policy to get more reward

**Key Concepts:**
- **Goal:** maximize total reward over time
- **Policy (π):** the method that converts states into actions
- **Reward (r):** the feedback returned by the environment
- **Exploration vs. Exploitation:** trying new things vs. using what is already known to work well

**Where is RL used?** Games, robotics and control, autonomous vehicles, finance and investment, medicine and public health

**Comparison with Other Methods (does it require a predefined answer?):**
| Method | Requires a Predefined Answer? |
|---|---|
| Supervised Learning | ✅ Yes |
| Unsupervised Learning | ❌ No |
| Reinforcement Learning | ❌ No (uses reward as the guiding signal) |

**Things to Remember:** No predefined answer / learns by taking action / receives rewards or penalties as outcomes / gradually improves over time

**Quick Summary:** RL learns by acting within an environment, trying many approaches, receiving rewards, and gradually improving to achieve the best possible outcome.

---

## 📌 Comparative Summary Table — Types of ML by Example

| Example | Type of ML | Input | Output |
|---|---|---|---|
| Spam email filter | Supervised Learning | Email text | Spam or Not Spam |
| Face unlock | Supervised Learning | Face image | Match or No Match |
| Product recommendation | Unsupervised Learning | What you clicked or bought | Items you may like |
| Music recommendation | Reinforcement Learning | Songs you listen to | Next songs to play |
| Self-driving car | Reinforcement Learning | Camera and sensor data | Turn, stop, go |
| Fraud detection | Supervised Learning | Payment history | Normal or Fraud |
| House price prediction | Supervised Learning | House size, location | Predicted price |
| Grouping customers | Unsupervised Learning | Age, purchase history | Group or segment |
| Game AI (chess, Go) | Reinforcement Learning | Game state | Best move |

---

## ✅ Summary 

**Conclusion**
- We saw the differences between the types of learning in ML
- We learned how ML is applied to different types of work today

**ML Challenge**
- A lightweight model for mobile phones?
- A model that works well with limited training data?

**Key Takeaways**
- **Understand:** We understand the differences between AI, ML, DL
- **Learn:** We learned how ML is used in many real-world applications
- **Challenge:** We face real challenges in building practical ML models
- **Improve:** We continue to learn, adapt, and improve our models
  
---























