import pandas as pd

train_df = pd.read_csv("/Users/robin/Desktop/my_projects/buyerforecaster/artifacts/train.csv")
print("Train data loaded successfully")
test_df = pd.read_csv("/Users/robin/Desktop/my_projects/buyerforecaster/artifacts/test.csv")
print("Test data loaded successfully")

print(f"Shape of train_df: {train_df.shape}")
print(f"Shape of test_df: {test_df.shape}")
