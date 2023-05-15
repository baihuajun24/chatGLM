import pandas as pd

# Load data from JSON file
df = pd.read_json('classification/data/train.json', lines=True)

# Print the data
print(df)

# Assuming df is your DataFrame and "subject_name" is your column
all_subjects = df['subject_name'].explode().unique()

print(all_subjects)

with open('classification/data/label.txt', 'w') as f:
    for item in all_subjects:
        f.write("%s\n" % item)