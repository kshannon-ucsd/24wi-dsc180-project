# %% [markdown]
# # Model 1 Evaluation
# 
# ### Importing Dependenceies

# %%
import os
import keras
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ### Loading Model 1

# %%
model = keras.saving.load_model("../notebooks/final_model_1.keras")

# %% [markdown]
# ### Loading Test Data

# %%
test_datagen = ImageDataGenerator()
test_df = {'filename': [], 'label': [], 'binary_label': []}

for file in os.listdir('../../data/processed/test'):
    if file.endswith('.jpeg'):
        if 'virus' in file or 'bacteria' in file:
            test_df['label'].append('1')
            test_df['binary_label'].append(True)
        else:
            test_df['label'].append('0')
            test_df['binary_label'].append(False)
            
        test_df['filename'].append(file) 

test_df = pd.DataFrame(test_df)

display(test_df)

test_generator = test_datagen.flow_from_dataframe(
    test_df, directory='../../data/processed/test',
    x_col='filename', 
    y_col='label', 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='binary', 
    validate_filenames=False,
    shuffle=False)

# %%
true_labels = test_generator.classes  # True class labels

# %%
y_pred = model.predict(test_generator)

y_pred_binary = (y_pred > 0.5).astype(int)  # Threshold at 0.5 for binary classification

save_dir = "../../plots"

# %%
report = classification_report(true_labels, y_pred_binary)
print(report)

with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# %%
cm = confusion_matrix(true_labels, y_pred_binary)

# Plotting the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(save_dir, "model_1_confusion_matrix.png"))  # Save the figure
plt.show()

# %%
fpr, tpr, threshold = roc_curve(true_labels, y_pred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(os.path.join(save_dir, "model_1_roc_curve.png"))  # Save the figure
plt.show()
plt.close()


