from huggingface_hub import dataset_info

info = dataset_info("ricl-vla/collected_demos_training")
# print(info.siblings)
print(sum(f.size for f in info.siblings if f.size is not None))
