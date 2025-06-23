# -Decision-Tree-Regressor-from-scratch


# 🌳 Decision Tree Regressor from Scratch (NumPy)

This project implements a **Decision Tree Regressor** using only NumPy and Matplotlib. It trains a decision tree on synthetic data and visualizes how well it fits the data using piecewise constant functions.

---

## 🧪 Data Generation

We generate a simple linear dataset with some random noise:
```python
x_train = 2 * np.random.rand(100, 1)
y_train = 4 + 3 * x_train + np.random.randn(100, 1)
```
- `x_train`: 100 random samples from 0 to 2
- `y_train`: Line `y = 3x + 4` with Gaussian noise added

The data is visualized using a scatter plot.

---

## 🧠 Code Explanation

### `variance(y)`
Computes the variance (spread) of values in `y`, used to measure impurity.
```python
def variance(y):
    if len(y)==0:
        return 0
    return np.mean((y - np.mean(y)) ** 2)
```

### `best_node(x, y)`
Finds the best split threshold in `x` that minimizes the **weighted variance** of `y` after the split.
```python
def best_node(x, y):
    best_score = float("inf")
    best_thresh = None
    for i in np.unique(x):
        left_y = y[x <= i]
        right_y = y[x > i]
        score = variance(left_y)*len(left_y) + variance(right_y)*len(right_y)
        if score < best_score:
            best_score = score
            best_thresh = i
    return best_thresh
```

### `decision_tree(x, y, d=0, m_d=3)`
Recursively builds the decision tree:
- Stops when max depth is reached or output is pure.
- Splits data at best threshold and builds left/right subtrees.
```python
def decision_tree(x, y, d=0, m_d=3):
    if d == m_d or len(np.unique(y)) == 1:
        return np.mean(y)  # leaf node
    threshold = best_node(x, y)
    if threshold is None:
        return np.mean(y)
    left = x <= threshold
    right = x > threshold
    left_tree = decision_tree(x[left], y[left], d+1, m_d)
    right_tree = decision_tree(x[right], y[right], d+1, m_d)
    return threshold, left_tree, right_tree
```

### `predict(tree, x_v)`
Traverses the tree to return prediction for a single value `x_v`.
```python
def predict(tree, x_v):
    if not isinstance(tree, tuple):
        return tree
    threshold, left_tree, right_tree = tree
    if x_v <= threshold:
        return predict(left_tree, x_v)
    else:
        return predict(right_tree, x_v)
```

---

## 📊 Visualization

We generate predictions on test points and plot the result:
```python
x_test = np.linspace(min(x_train)-1, max(x_train)+1, 500)
y_pred = [predict(tree, i) for i in x_test]
```
Then we plot the training data and predicted curve:
```python
plt.scatter(x_train, y_train, color='blue')
plt.plot(x_test, y_pred, color='red')
```

### Output:
- **Blue points**: training data
- **Red line**: piecewise constant regression from the tree

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/your-username/decision-tree-regressor-scratch.git
cd decision-tree-regressor-scratch
```

2. Install required packages:
```bash
pip install numpy matplotlib
```

3. Run the script:
```bash
python decision_tree_regression.py
```

---

## 📁 File Structure

```
decision-tree-regressor-scratch/
│
├── decision_tree_regression.py  # Main script
├── plot_example.png             # Optional: saved output image
└── README.md                    # Project documentation
```

---

## 👤 Author

**Mudit Gahlot**  
Open-source project for learning and experimentation.

---

## 📜 License

This project is licensed under the MIT License.
