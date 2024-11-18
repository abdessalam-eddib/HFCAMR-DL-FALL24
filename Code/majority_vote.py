import pandas as pd

# Define the threshold for majority voting
threshold = 1

# Load the CSV files  # We vary the selection of agents to achieve highest performance
df_0 = pd.read_csv("results_0.csv")
df_1 = pd.read_csv("results_1.csv")
df_2 = pd.read_csv("results_2.csv")
df_3 = pd.read_csv("results_3.csv")
df_4 = pd.read_csv("results_4.csv")
df_5 = pd.read_csv("results_5.csv")
df_6 = pd.read_csv("results_6.csv")
df_7 = pd.read_csv("results_7.csv")
df_8 = pd.read_csv("results_8.csv")
df_9 = pd.read_csv("results_9.csv")
df_10 = pd.read_csv("results_10.csv")
df_11 = pd.read_csv("results_11.csv")
df_12 = pd.read_csv("results_12.csv")
df_13 = pd.read_csv("results_13.csv")
df_14 = pd.read_csv("results_14.csv")
df_15 = pd.read_csv("results_15.csv")
df_16 = pd.read_csv("results_16.csv")
df_17 = pd.read_csv("results_17.csv")
df_18 = pd.read_csv("results_18.csv")
df_19 = pd.read_csv("results_19.csv")
df_20 = pd.read_csv("results_20.csv")
df_21 = pd.read_csv("results_21.csv")
df_22 = pd.read_csv("results_22.csv")
df_23 = pd.read_csv("results_23.csv")
df_24 = pd.read_csv("results_24.csv")
df_25 = pd.read_csv("results_25.csv")
df_26 = pd.read_csv("results_26.csv")
df_27 = pd.read_csv("results_27.csv")
df_28 = pd.read_csv("results_28.csv")
df_29 = pd.read_csv("results_29.csv")
df_30 = pd.read_csv("results_30.csv")
df_31 = pd.read_csv("results_31.csv")
df_32 = pd.read_csv("results_32.csv")
df_33 = pd.read_csv("results_33.csv")
df_34 = pd.read_csv("results_34.csv")


# Convert to string
df_0["is_correct"] = df_0["is_correct"].astype(str)
df_1["is_correct"] = df_1["is_correct"].astype(str)
df_2["is_correct"] = df_2["is_correct"].astype(str)
df_3["is_correct"] = df_3["is_correct"].astype(str)
df_4["is_correct"] = df_4["is_correct"].astype(str)
df_5["is_correct"] = df_5["is_correct"].astype(str)
df_6["is_correct"] = df_6["is_correct"].astype(str)
df_7["is_correct"] = df_7["is_correct"].astype(str)
df_8["is_correct"] = df_8["is_correct"].astype(str)
df_9["is_correct"] = df_9["is_correct"].astype(str)
df_10["is_correct"] = df_10["is_correct"].astype(str)
df_11["is_correct"] = df_11["is_correct"].astype(str)
df_12["is_correct"] = df_12["is_correct"].astype(str)
df_13["is_correct"] = df_13["is_correct"].astype(str)
df_14["is_correct"] = df_14["is_correct"].astype(str)
df_15["is_correct"] = df_15["is_correct"].astype(str)
df_16["is_correct"] = df_16["is_correct"].astype(str)
df_17["is_correct"] = df_17["is_correct"].astype(str)
df_18["is_correct"] = df_18["is_correct"].astype(str)
df_19["is_correct"] = df_19["is_correct"].astype(str)
df_20["is_correct"] = df_20["is_correct"].astype(str)
df_21["is_correct"] = df_21["is_correct"].astype(str)
df_22["is_correct"] = df_22["is_correct"].astype(str)
df_23["is_correct"] = df_23["is_correct"].astype(str)
df_24["is_correct"] = df_24["is_correct"].astype(str)
df_25["is_correct"] = df_25["is_correct"].astype(str)
df_26["is_correct"] = df_26["is_correct"].astype(str)
df_27["is_correct"] = df_27["is_correct"].astype(str)
df_28["is_correct"] = df_28["is_correct"].astype(str)
df_29["is_correct"] = df_29["is_correct"].astype(str)
df_30["is_correct"] = df_30["is_correct"].astype(str)
df_31["is_correct"] = df_31["is_correct"].astype(str)
df_32["is_correct"] = df_32["is_correct"].astype(str)
df_33["is_correct"] = df_33["is_correct"].astype(str)
df_34["is_correct"] = df_34["is_correct"].astype(str)


# Merge the dataframes
df = pd.DataFrame({
    "ID": range(len(df_1)),
    "is_correct_0": df_0["is_correct"],
    "is_correct_1": df_1["is_correct"],
    "is_correct_2": df_2["is_correct"],
    "is_correct_3": df_3["is_correct"],
    "is_correct_4": df_4["is_correct"],
    "is_correct_5": df_5["is_correct"],
    "is_correct_6": df_6["is_correct"],
    "is_correct_7": df_7["is_correct"],
    "is_correct_8": df_8["is_correct"],
    "is_correct_9": df_9["is_correct"],
    "is_correct_10": df_10["is_correct"],
    "is_correct_11": df_11["is_correct"],
    "is_correct_12": df_12["is_correct"],
    "is_correct_13": df_13["is_correct"],
    "is_correct_14": df_14["is_correct"],
    "is_correct_15": df_15["is_correct"],
    "is_correct_16": df_16["is_correct"],
    "is_correct_17": df_17["is_correct"],
    "is_correct_18": df_18["is_correct"],
    "is_correct_19": df_19["is_correct"],
    "is_correct_20": df_20["is_correct"],
    "is_correct_21": df_21["is_correct"],
    "is_correct_22": df_22["is_correct"],
    "is_correct_23": df_23["is_correct"],
    "is_correct_24": df_24["is_correct"],
    "is_correct_26": df_26["is_correct"],
    "is_correct_27": df_27["is_correct"],
    "is_correct_28": df_28["is_correct"],
    "is_correct_29": df_29["is_correct"],
    "is_correct_31": df_31["is_correct"],
    "is_correct_32": df_32["is_correct"],
    "is_correct_33": df_33["is_correct"],
    "is_correct_30": df_30["is_correct"],
})

# Combine the results  using majority voting, but give priority to True values
df["is_correct"] = df.apply(lambda x: "True" if x.str.count("True").sum() > threshold else "False", axis=1)

# Delete the intermediate columns
df = df.drop(columns=[f"is_correct_{i}" for i in range(34)])

# Check the final distribution of True and False values
true_count = df["is_correct"].str.count("True").sum()
false_count = df["is_correct"].str.count("False").sum()

print(f"Number of True values: {true_count}")
print(f"Number of False values: {false_count}")

# Save the results
df.to_csv("majority_vote.csv", index=False)