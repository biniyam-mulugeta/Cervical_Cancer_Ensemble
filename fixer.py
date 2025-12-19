import pandas as pd

file_path = 'FinetunedResNet50_predictions.csv'
df = pd.read_csv(file_path)
df['image_name'] = df['image_name'].astype(str) + '.png'
df.to_csv(file_path, index=False)

print("Successfully fixed FinetunedResNet50_predictions.csv")