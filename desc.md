Based on the diagram you provided and our previous discussions on the DAD (Differentiable Adaptive Discretization) architecture, here is a formal description of each module, its inputs/outputs, and the data flow.

You can use this text for your system architecture section.

---

### **System Architecture Overview**

The proposed framework is a **Generator-Teacher-Student** architecture designed for automated t-way testing. The system simultaneously learns an optimal discretization scheme (Bin Learner) and generates synthetic test cases (Generator) that maximize interaction coverage over those learned bins.

#### **1. The Generator Module**
* **Description:** A neural network responsible for synthesizing continuous test inputs. It learns to explore the feature space to satisfy the diversity requirements of t-way testing.
* **Input:** A random latent vector $z \sim \mathcal{N}(0, I)$ (Gaussian noise).
* **Output:** A batch of synthetic continuous feature vectors $X_{gen} \in \mathbb{R}^{N \times M}$, where $N$ is batch size and $M$ is the number of features.
* **Data Flow:** The generated features $X_{gen}$ are broadcast to three downstream components: the **Teacher**, the **Bin Learner**, and the **Student**.
* **Training Signal:** It receives gradients from the **Interaction Diversity Loss**, encouraging it to generate diverse samples that cover rare bin interactions.

#### **2. The Teacher Module (The Oracle)**
* **Description:** A pre-trained, frozen high-performance model (e.g., Gradient Boosting or Deep Neural Network) that encapsulates the "Dark Knowledge" of the system under test. It serves as the ground truth for system behavior.
* **Input:** The synthetic features $X_{gen}$ from the Generator.
* **Output:** The prediction probability distribution $Y_{teacher}$ (Softmax output or regression values).
* **Data Flow:** The prediction distribution is sent to the **Bin Learner** to calculate the semantic consistency (dispersion) of the proposed bins.

#### **3. The Bin Learner (DAD Module)**
* **Description:** The **Differentiable Adaptive Discretization** module. It learns the optimal boundary locations for each feature. Instead of static equal-width binning, it dynamically adjusts bin widths to group input regions where the Teacher's predictions are consistent.
* **Input:**
    1.  Features $X_{gen}$ (to calculate bin membership).
    2.  Teacher Predictions $Y_{teacher}$ (to calculate bin purity).
* **Output:**
    1.  **Learned Bins:** The differentiable bin boundaries $\beta$.
    2.  **Soft Membership Matrix:** A tensor representing the probability of each sample belonging to each bin.
* **Loss Mechanism:** It computes the **Dispersion Loss**. This loss is backpropagated *only* to the Bin Learner parameters to adjust boundaries. It ensures that within any given bin, the variance of the Teacher's predictions is minimized.

#### **4. The Student (Distillation & Coverage Module)**
* **Description:** This module acts as the bridge between the continuous generation and the discrete t-way objective. It applies the **Learned Bins** (from the Bin Learner) to the continuous output of the Generator to determine the discrete "state" of each test case.
* **Input:**
    1.  Features $X_{gen}$ from the Generator.
    2.  **Learned Bins** parameters from the Bin Learner.
* **Output:** Discrete interaction vectors (softly discretized).
* **Loss Mechanism:** It computes the **Interaction Diversity Loss** (Interaction Div Loss). This loss measures how many unique t-way tuples (pairs, triples) are covered by the current batch.
* **Data Flow:** The gradients from this loss are backpropagated through the Student to the **Generator**, forcing the Generator to produce data points that fill "missing" interactions in the learned bin space.

---

### **Data Flow Summary**

1.  **Synthesis:** The **Generator** produces a batch of synthetic features ($X$).
2.  **Ground Truth:** The **Teacher** predicts the behavior ($Y$) for these features.
3.  **Discretization Learning:** The **Bin Learner** takes $X$ and $Y$. It adjusts its internal boundaries to minimize **Dispersion Loss** (making bins "pure" regarding $Y$).
4.  **Coverage Optimization:** The **Student** takes $X$ and applies the boundaries learned in step 3. It calculates the **Interaction Diversity Loss** (checking if all t-way combinations of bins are present).
5.  **Feedback:** The Diversity Loss updates the **Generator**, teaching it to create test cases that cover the unexplored combinations defined by the Bin Learner.