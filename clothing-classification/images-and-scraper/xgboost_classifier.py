import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from tqdm import tqdm
from skimage.feature import canny


def load_images_from_folder(base_path, target_size=(224, 224), bins=32):

    images = []
    labels = []

    # Main categories
    for main_category in os.listdir(base_path):
        main_category_path = os.path.join(base_path, main_category)
        if not os.path.isdir(main_category_path):
            continue

        # Sub categories
        for sub_category in os.listdir(main_category_path):
            sub_category_path = os.path.join(main_category_path, sub_category)
            if not os.path.isdir(sub_category_path):
                continue

            print(f"Loading images from {main_category}/{sub_category}")

            # Images in sub category
            for img_name in os.listdir(sub_category_path):
                if not any(img_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    continue

                img_path = os.path.join(sub_category_path, img_name)
                try:
                    # Load and resize the image
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(target_size)
                    img_array = np.array(img)

                    # ---- Feature 1: Flattened Pixel Values ----
                    pixel_values = img_array.flatten()

                    # ---- Feature 2: Color Histogram (RGB) ----
                    # Compute histograms for each channel (R, G, B)
                    r_hist = np.histogram(img_array[:, :, 0], bins=bins, range=(0, 256), density=True)[0]
                    g_hist = np.histogram(img_array[:, :, 1], bins=bins, range=(0, 256), density=True)[0]
                    b_hist = np.histogram(img_array[:, :, 2], bins=bins, range=(0, 256), density=True)[0]

                    # Combine all histograms into a single vector
                    color_histogram = np.hstack([r_hist, g_hist, b_hist])

                    # ---- Feature 3: Edge Detection (Canny) ----
                    # Convert image to grayscale
                    gray_img = np.array(img.convert('L'))

                    # Perform Canny edge detection
                    edges = canny(gray_img, sigma=1.0)  # Adjust sigma for sensitivity

                    # Flatten edge map
                    edge_features = edges.flatten()

                    # ---- Combine Features ----
                    combined_features = np.hstack([pixel_values, color_histogram, edge_features])

                    # Append the features and label
                    images.append(combined_features)
                    labels.append(sub_category)

                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    if not images:
        raise ValueError("No images were loaded. Please check the path and image formats.")

    return np.array(images), np.array(labels)



def plot_confusion_matrix(y_true, y_pred, label_encoder):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()



def apply_pca(X, explained_variance):
    pca = PCA(n_components=explained_variance)
    X_reduced = pca.fit_transform(X)
    print(f"Original shape: {X.shape}")
    print(f"Reduced shape: {X_reduced.shape}")
    print(f"Explained variance ratio (sum): {np.sum(pca.explained_variance_ratio_):.4f}")
    return X_reduced, pca



def k_fold_evaluation(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=5,
            gamma=0.4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
        )

        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
        print(f"Fold {fold} Accuracy: {score:.4f}")

    print(f"\nAverage K-Fold Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores


def bootstrap_evaluation(X, y, n_iterations=100):
    scores = []

    for i in tqdm(range(n_iterations), desc="Bootstrap Progress"):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]

        X_train, X_test, y_train, y_test = train_test_split(
            X_bootstrap, y_bootstrap, test_size=0.2, random_state=i
        )

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=5,
            gamma=0.4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)

    print(f"\nBootstrap Results:")
    print(f"Mean Accuracy: {np.mean(scores):.4f}")
    print(f"95% Confidence Interval: ({np.percentile(scores, 2.5):.4f}, {np.percentile(scores, 97.5):.4f})")
    return scores



def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to your image folders
    base_path = os.path.join(current_dir, "clothing_images")

    print(f"Loading images from: {base_path}")

    # Load and preprocess images
    print("Extracting features from images...")
    X, y = load_images_from_folder(base_path)
    print(f"Extracted features from {len(X)} images with {len(np.unique(y))} different classes")

    print("\nApplying PCA to reduce dimensionality...")
    X, pca_model = apply_pca(X, explained_variance=0.90)  # Preserve 90% variance

    # Encode string labels to numerical labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print("\nClass mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {i}")

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% train, 30% temp
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% val, 15% test

    # Train XGBoost model
    print("\nTraining XGBoost model...")

    model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=3,
        eval_metric='mlogloss',
        objective='multi:softprob',  # Multi-class classification
        num_class=len(np.unique(y)),
        learning_rate=0.01,
        min_child_weight=13,
        gamma=0.4,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10
    )

    # Plot training progress
    results = model.evals_result()
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train Loss')
    plt.plot(x_axis, results['validation_1']['mlogloss'], label='Validation Loss')
    plt.title('XGBoost Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Make predictions
    y_pred = model.predict(X_test)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, label_encoder)

    # Perform K-Fold evaluation
    print("\nPerforming K-Fold Cross Validation...")
    k_fold_scores = k_fold_evaluation(X, y)

    # Perform Bootstrap evaluation
    print("\nPerforming Bootstrap Evaluation...")
    bootstrap_scores = bootstrap_evaluation(X, y)


if __name__ == "__main__":
    main()
