from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
import os


print("Veri seti yükleniyor...")
dataset = load_dataset("oscar", "unshuffled_deduplicated_tr", split="train[:1%]")  # Daha büyük oranla değiştirilebilir


print("Tokenizer eğitiliyor...")
tokenizer_model = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer_model.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
tokenizer_model.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.WordLevelTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

def batch_iterator():
    for i in range(0, len(dataset), 1000):
        yield dataset[i:i+1000]["text"]

tokenizer_model.train_from_iterator(batch_iterator(), trainer)
tokenizer_model.save("tokenizer.json")


tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json",
                                    unk_token="[UNK]", pad_token="[PAD]",
                                    cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


print("Model oluşturuluyor...")
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    max_position_embeddings=512,
    type_vocab_size=1
)

model = BertForMaskedLM(config)


print("Veri tokenize ediliyor...")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./bert-from-scratch-tr",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    logging_dir="./logs",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("Eğitim başlıyor...")
trainer.train()

# Save
print("Model ve tokenizer kaydediliyor...")
trainer.save_model("./bert-from-scratch-tr")
tokenizer.save_pretrained("./bert-from-scratch-tr")

print("✅ Eğitim tamamlandı!")
