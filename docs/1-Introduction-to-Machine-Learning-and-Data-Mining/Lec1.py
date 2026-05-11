from datasets import load_dataset
import pandas as pd
from collections import Counter
import re

dataset = load_dataset("nvidia/Nemotron-Personas-Korea", split="train[:100]")

df = dataset.to_pandas()
df.head()

df["professional_persona"].head(10)

all_text = " ".join(df["professional_persona"].tolist())
words = re.findall(r"[가-힣]{2,}\b", all_text)

word_counts = Counter(words)
print(word_counts.most_common(20))
