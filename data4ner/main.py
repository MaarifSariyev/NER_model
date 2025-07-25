from datasets import load_dataset

dataset = load_dataset("json", data_files="kyc_dataset.json", field=None)
print(dataset["train"][0])