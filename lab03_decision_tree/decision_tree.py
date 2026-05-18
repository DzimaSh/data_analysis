# -*- coding: utf-8 -*-
import numpy as np

class Node:
    """
    Клас для прадстаўлення вузла расшчаплення ў дрэве рашэнняў.
    """
    def __init__(self, index, t, true_branch, false_branch):
        self.index = index          # Індэкс прыкметы, па якой адбываецца расшчапленне
        self.t = t                  # Парог расшчаплення
        self.true_branch = true_branch    # Левае паддрэва (аб'екты, для якіх прыкмета <= t)
        self.false_branch = false_branch  # Правае паддрэва (аб'екты, для якіх прыкмета > t)

class Leaf:
    """
    Клас для прадстаўлення ліста (канчатковага вузла) у дрэве рашэнняў.
    """
    def __init__(self, data, labels):
        self.data = data            # Аб'екты, якія трапілі ў гэты ліст
        self.labels = labels        # Пазнакі класаў для гэтых аб'ектаў
        self.prediction = self.predict()  # Прагназуемы клас для гэтага ліста

    def predict(self):
        """
        Вылічвае прагноз для ліста на аснове мажарытарнага голасу (majority vote).
        """
        classes, counts = np.unique(self.labels, return_counts=True)
        if len(counts) == 0:
            return None
        return classes[np.argmax(counts)]

def gini(labels):
    """
    Вылічвае крытэрый Джыні (Gini impurity) для набору пазнак класаў.
    """
    labels = np.array(labels)
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return 1.0 - np.sum(probabilities ** 2)

def gain(left_labels, right_labels, root_gini):
    """
    Вылічвае прырост інфармацыі (Information Gain) пасля расшчаплення.
    """
    n_left = len(left_labels)
    n_right = len(right_labels)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    
    p_left = n_left / n_total
    p_right = n_right / n_total
    
    return root_gini - (p_left * gini(left_labels) + p_right * gini(right_labels))

def split(data, labels, column_index, t):
    """
    Расшчапляе датасет на левую (<= t) і правую (> t) часткі.
    """
    data = np.array(data)
    labels = np.array(labels)
    left_mask = data[:, column_index] <= t
    right_mask = ~left_mask
    return data[left_mask], labels[left_mask], data[right_mask], labels[right_mask]

def find_best_split(data, labels, min_samples_leaf=1):
    """
    Знаходзіць найлепшае расшчапленне (прыкмету і парог), якое максімізуе Information Gain.
    """
    root_gini = gini(labels)
    best_gain = 0.0
    best_t = None
    best_index = None
    
    n_samples, n_features = data.shape
    
    for col in range(n_features):
        unique_values = np.unique(data[:, col])
        # Выкарыстоўваем сярэднія кропкі паміж суседнімі значэннямі прыкметы для больш гладкага падзелу
        if len(unique_values) > 1:
            sorted_vals = np.sort(unique_values)
            thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
        else:
            thresholds = unique_values
            
        for val in thresholds:
            left_data, left_labels, right_data, right_labels = split(data, labels, col, val)
            
            # Крытэрый прыпынку: мінімальная колькасць аб'ектаў у лісце
            if len(left_labels) < min_samples_leaf or len(right_labels) < min_samples_leaf:
                continue
                
            current_gain = gain(left_labels, right_labels, root_gini)
            
            if current_gain > best_gain:
                best_gain = current_gain
                best_t = val
                best_index = col
                
    return best_gain, best_index, best_t

def build_tree(data, labels, max_depth=None, min_samples_leaf=1, current_depth=0):
    """
    Рэкурсіўна будуе дрэва рашэнняў.
    """
    data = np.array(data)
    labels = np.array(labels)
    
    # Калі няма аб'ектаў
    if len(labels) == 0:
        return Leaf(data, labels)
        
    # Базавыя выпадкі прыпынку:
    # 1. Усе аб'екты маюць адзін клас
    if len(np.unique(labels)) == 1:
        return Leaf(data, labels)
        
    # 2. Дасягнута максімальная глыбіня
    if max_depth is not None and current_depth >= max_depth:
        return Leaf(data, labels)
        
    # 3. Колькасць аб'ектаў меншая або роўная min_samples_leaf
    if len(labels) <= min_samples_leaf:
        return Leaf(data, labels)
        
    # Пошук найлепшага падзелу
    best_gain, best_index, best_t = find_best_split(data, labels, min_samples_leaf)
    
    # Калі няма карыснага падзелу (Gain блізкі да 0 ці няма магчымасці раздзяліць)
    if best_gain <= 1e-7 or best_index is None:
        return Leaf(data, labels)
        
    # Расшчапленне
    left_data, left_labels, right_data, right_labels = split(data, labels, best_index, best_t)
    
    # Будаўніцтва веткаў
    true_branch = build_tree(left_data, left_labels, max_depth, min_samples_leaf, current_depth + 1)
    false_branch = build_tree(right_data, right_labels, max_depth, min_samples_leaf, current_depth + 1)
    
    return Node(best_index, best_t, true_branch, false_branch)

def classify_object(obj, node):
    """
    Рэкурсіўна класіфікуе адзін аб'ект, праходзячы па дрэве.
    """
    if isinstance(node, Leaf):
        return node.prediction
        
    if obj[node.index] <= node.t:
        return classify_object(obj, node.true_branch)
    else:
        return classify_object(obj, node.false_branch)

def predict(data, tree):
    """
    Прадказвае класы для ўсяго датасэта.
    """
    data = np.array(data)
    return np.array([classify_object(obj, tree) for obj in data])

def print_tree(node, spacing=""):
    """
    Рэкурсіўна выводзіць структуру дрэва рашэнняў у тэкставым выглядзе.
    """
    if isinstance(node, Leaf):
        print(spacing + f"[Ліст] Прагноз = {node.prediction} (аб'ектаў: {len(node.labels)})")
        return
        
    print(spacing + f"[Вузел] Прыкмета {node.index} <= {node.t:.4f}")
    print(spacing + "  |-- ТАК:")
    print_tree(node.true_branch, spacing + "  |   ")
    print(spacing + "  L-- НЕ:")
    print_tree(node.false_branch, spacing + "      ")

def accuracy_metric(actual, predicted):
    """
    Вылічвае дакладнасць (Accuracy) прадказанняў.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    if len(actual) == 0:
        return 0.0
    return np.mean(actual == predicted)

class CustomDecisionTreeClassifier:
    """
    Клас-абгортка для сумяшчальнасці з інтэрфейсам scikit-learn.
    """
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.tree_ = build_tree(X, y, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        return self

    def predict(self, X):
        X = np.array(X)
        if self.tree_ is None:
            raise ValueError("Мадэль яшчэ не навучана. Спачатку выклічце fit().")
        return predict(X, self.tree_)

    def print_tree(self):
        if self.tree_ is None:
            print("Дрэва яшчэ не пабудавана. Выклічце fit() спачатку.")
        else:
            print_tree(self.tree_)
