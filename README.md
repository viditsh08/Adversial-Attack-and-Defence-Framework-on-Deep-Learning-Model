# 🛡️ Adversarial Attack & Defense Framework

**Python | PyTorch | Deep Learning | AI Security**

A deep learning-based framework designed to evaluate model vulnerabilities using adversarial attack techniques and improve robustness through defense strategies.

---

## 📌 Project Overview

This project focuses on analyzing the security of deep learning models by applying adversarial attacks that introduce small perturbations in input data to mislead the model. It also implements defense mechanisms to improve model robustness and reliability.

---

## 🎯 Features

* ⚡ Implementation of adversarial attack (FGSM)
* 🧠 Model evaluation under adversarial conditions
* 🛡️ Defense using adversarial training
* 📊 Visualization of model performance
* 📉 Comparison of accuracy before and after attack

---

## 🧠 Working Principle

* Train a deep learning model
* Generate adversarial samples using **FGSM**
* Evaluate model predictions on attacked data
* Apply defense techniques
* Compare performance before and after defense

---

## ⚙️ Adversarial Attack (FGSM)

```
x_adv = x + ε * sign(∇x J(θ, x, y))
```

---

## 🛠️ Tech Stack

* Python 🐍
* PyTorch 🔥
* NumPy 🔢
* Matplotlib 📊
* Seaborn 🎨

---

## 📁 Project Structure

```
Adversarial-Attack-Defense/
│
├── src/
│   └── main.py
├── models/
├── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Adversarial-Attack-Defense.git
cd Adversarial-Attack-Defense
```

### 2️⃣ Install Dependencies

```bash
pip install torch numpy matplotlib seaborn
```

---

## ▶️ Run the Project

```bash
python src/main.py
```

---

## 📸 Screenshots

## 📸 Screenshots

### 🔹 Original and Adversarial Prediction on Baseline Model

![Baseline](images/baseline_predictions.png)

### 🔹 Original and Adversarial Prediction on Defense Model

![Defense](images/defense_predictions.png)

### 🔹 Performance Comparison Graph

![Performance](images/performance_graph.png)


---

## 📊 Results

* Significant drop in model accuracy after applying adversarial attack
* Misclassification observed even with small perturbations
* Improved performance after applying adversarial training
* Visualization clearly shows difference between original and attacked predictions

---

## 🎓 Applications

* AI model security 🔐
* Robust machine learning systems 🤖
* Research in adversarial attacks 📊

---

## 🚀 Future Improvements

* Add advanced attacks (PGD, CW)
* Improve defense strategies
* Use larger datasets

---

## 🤝 Contribution

Contributions are welcome!

---

## 📄 License

Open-source project under MIT License.

---

## 🙌 Author

**Vidit Sharma**
Computer Science (AI/ML) Student

---

⭐ Star the repository if you found it useful!
