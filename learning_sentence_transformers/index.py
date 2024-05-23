# Load model directly
from transformers import AutoTokenizer, TFAutoModel


MINI_LM = "sentence-transformers/paraphrase-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MINI_LM)
model = TFAutoModel.from_pretrained(MINI_LM)

vocab_wrd2idx = tokenizer.vocab
vocab_idx2wrd = {v: k for k, v in vocab_wrd2idx.items()}

model_weights = model.get_weights()
print(model_weights)
# vocab_weights = model_weights[0]