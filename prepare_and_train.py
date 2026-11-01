# =========================================================
# Brain Tumor MRI Classification
# MobileNetV2 Transfer Learning
# Feature Extraction + Fine-Tuning + Evaluation
# =========================================================

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
    RandomFlip,
    RandomContrast,
    GlobalAveragePooling2D
)


# =========================
# 1) الإعدادات الأساسية
# =========================

img_size = (224, 224)
batch_size = 16

data_dir_train = r"C:\Users\xayn\Desktop\archive\Training"
data_dir_test = r"C:\Users\xayn\Desktop\archive\Testing"

models_dir = r"C:\Users\xayn\Desktop\MRI classification\models"
results_dir = r"C:\Users\xayn\Desktop\MRI classification\results"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


# =========================
# 2) تقسيم البيانات
# =========================

# 80% تدريب و20% تحقق من مجلد Training
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_train,
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2,
    subset="training"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_train,
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2,
    subset="validation"
)

# Testing يبقى للاختبار النهائي فقط
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_test,
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Training classes:", train_ds.class_names)
print("Validation classes:", val_ds.class_names)
print("Testing classes:", test_ds.class_names)
print("Number of classes:", num_classes)

assert train_ds.class_names == val_ds.class_names == test_ds.class_names, (
    "Class names or class order differ between datasets"
)


# =========================
# 3) عرض أول باتش من الصور
# =========================

for images, labels in train_ds.take(1):
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    plt.figure(figsize=(10, 10))

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

    plt.savefig(
        os.path.join(results_dir, "sample_images.png"),
        bbox_inches="tight"
    )

plt.show()


# =========================
# 4) فحص قيم البكسلات
# =========================

for images, labels in train_ds.take(1):
    print("Data type:", images.dtype)
    print("Minimum:", tf.reduce_min(images).numpy())
    print("Maximum:", tf.reduce_max(images).numpy())


# =========================
# 5) Data Augmentation والتطبيع
# =========================

data_augmentation = Sequential([
    RandomRotation(0.05),
    RandomZoom(0.08),
    RandomTranslation(0.05, 0.05),
    RandomFlip("horizontal"),
    RandomContrast(0.1)
])

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)

AUTOTUNE = tf.data.AUTOTUNE

train_ds_opt = train_ds.map(
    lambda x, y: (
        normalization_layer(data_augmentation(x, training=True)),
        tf.one_hot(y, depth=num_classes)
    )
).prefetch(buffer_size=AUTOTUNE)

val_ds_opt = val_ds.map(
    lambda x, y: (
        normalization_layer(x),
        tf.one_hot(y, depth=num_classes)
    )
).prefetch(buffer_size=AUTOTUNE)

test_ds_opt = test_ds.map(
    lambda x, y: (
        normalization_layer(x),
        tf.one_hot(y, depth=num_classes)
    )
).prefetch(buffer_size=AUTOTUNE)


# =========================
# 6) بناء موديل MobileNetV2
# =========================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


def create_transfer_model(num_classes):
    model = Sequential([
        tf.keras.Input(shape=(224, 224, 3)),

        base_model,

        GlobalAveragePooling2D(),

        Dense(64, activation="relu"),
        Dropout(0.4),

        Dense(num_classes, activation="softmax")
    ])

    return model


model = create_transfer_model(num_classes)

model.summary()


# =========================
# 7) تجهيز الموديل للمرحلة الأولى
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0002
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    ),
    metrics=["accuracy"]
)


# =========================
# 8) Callbacks للمرحلة الأولى
# =========================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

checkpoint_feature_extraction = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(models_dir, "best_model_feature_extraction.keras"),
    monitor="val_loss",
    save_best_only=True
)


# =========================
# 9) المرحلة الأولى: Feature Extraction
# =========================

print("\n==============================")
print("Phase 1: Feature Extraction")
print("==============================\n")

history_feature_extraction = model.fit(
    train_ds_opt,
    validation_data=val_ds_opt,
    epochs=25,
    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint_feature_extraction
    ]
)


# =========================
# 10) المرحلة الثانية: Fine-Tuning
# =========================

print("\n==============================")
print("Phase 2: Fine-Tuning")
print("==============================\n")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    ),
    metrics=["accuracy"]
)

early_stop_finetune = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

reduce_lr_finetune = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

checkpoint_finetune = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(models_dir, "best_model_finetuned.keras"),
    monitor="val_loss",
    save_best_only=True
)

history_finetuning = model.fit(
    train_ds_opt,
    validation_data=val_ds_opt,
    epochs=25,
    callbacks=[
        early_stop_finetune,
        reduce_lr_finetune,
        checkpoint_finetune
    ]
)


# =========================
# 11) رسم نتائج Feature Extraction
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    history_feature_extraction.history["accuracy"],
    label="Training Accuracy"
)
plt.plot(
    history_feature_extraction.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Feature Extraction Accuracy")

plt.savefig(
    os.path.join(results_dir, "feature_extraction_accuracy.png"),
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    history_feature_extraction.history["loss"],
    label="Training Loss"
)
plt.plot(
    history_feature_extraction.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Feature Extraction Loss")

plt.savefig(
    os.path.join(results_dir, "feature_extraction_loss.png"),
    bbox_inches="tight"
)

plt.show()


# =========================
# 12) رسم نتائج Fine-Tuning
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    history_finetuning.history["accuracy"],
    label="Training Accuracy"
)
plt.plot(
    history_finetuning.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Fine-Tuning Accuracy")

plt.savefig(
    os.path.join(results_dir, "fine_tuning_accuracy.png"),
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    history_finetuning.history["loss"],
    label="Training Loss"
)
plt.plot(
    history_finetuning.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Fine-Tuning Loss")

plt.savefig(
    os.path.join(results_dir, "fine_tuning_loss.png"),
    bbox_inches="tight"
)

plt.show()


# =========================
# 13) التقييم النهائي على Testing
# =========================

test_loss, test_accuracy = model.evaluate(test_ds_opt)

print("Final Test Loss:", test_loss)
print("Final Test Accuracy:", test_accuracy)


# =========================
# 14) التنبؤات على Testing
# =========================

y_true = []
y_pred = []

for images, labels in test_ds_opt:
    predictions = model.predict(images, verbose=0)

    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# Classification Report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

with open(
    os.path.join(results_dir, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# =========================
# 15) Confusion Matrix
# =========================

cm = confusion_matrix(y_true, y_pred)

print("Class order:", class_names)
print("Confusion matrix:")
print(cm)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix on Test Set")

plt.savefig(
    os.path.join(results_dir, "confusion_matrix.png"),
    bbox_inches="tight"
)

plt.show()


# =========================
# 16) Precision, Recall, F1-score لكل فئة
# =========================

precision, recall, f1_score, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=np.arange(num_classes)
)

plt.figure(figsize=(10, 6))

x = np.arange(num_classes)
width = 0.25

plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f1_score, width, label="F1-score")

plt.xticks(x, class_names, rotation=20)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Precision, Recall and F1-score per Class")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.savefig(
    os.path.join(results_dir, "precision_recall_f1.png"),
    bbox_inches="tight"
)

plt.show()


# =========================
# 17) Accuracy لكل فئة
# =========================

class_accuracy = []

for i in range(num_classes):
    correct = cm[i, i]
    total = cm[i].sum()

    if total > 0:
        class_accuracy.append(correct / total)
    else:
        class_accuracy.append(0)

plt.figure(figsize=(8, 5))

plt.bar(class_names, class_accuracy, color="steelblue")
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Accuracy per Class on Test Set")
plt.grid(axis="y", alpha=0.3)

for i, acc in enumerate(class_accuracy):
    plt.text(
        i,
        acc + 0.02,
        f"{acc:.2f}",
        ha="center"
    )

plt.savefig(
    os.path.join(results_dir, "accuracy_per_class.png"),
    bbox_inches="tight"
)

plt.show()


# =========================
# 18) ملخص نهائي
# =========================

best_val_loss_fe = min(history_feature_extraction.history["val_loss"])
best_val_acc_fe = max(history_feature_extraction.history["val_accuracy"])

best_val_loss_ft = min(history_finetuning.history["val_loss"])
best_val_acc_ft = max(history_finetuning.history["val_accuracy"])

print("\n==============================")
print("Feature Extraction Results")
print("==============================")
print("Best Validation Loss:", best_val_loss_fe)
print("Best Validation Accuracy:", best_val_acc_fe)

print("\n==============================")
print("Fine-Tuning Results")
print("==============================")
print("Best Validation Loss:", best_val_loss_ft)
print("Best Validation Accuracy:", best_val_acc_ft)

print("\n==============================")
print("Final Test Results")
print("==============================")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

print("\nBest Fine-Tuned Model saved at:")
print(os.path.join(models_dir, "best_model_finetuned.keras"))