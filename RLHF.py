import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,   
    GenerationConfig
)
from datasets import load_dataset
from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model


# trl: Transformer Reinforcement Learning library
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler

import torch
import evaluate

import numpy as np
import pandas as pd

# tqdm library makes the loops show a smart progress meter.
from tqdm import tqdm
tqdm.pandas()

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
huggingface_dataset_name = "allenai/real-toxicity-prompts"

dataset_original = load_dataset(huggingface_dataset_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dataset(model_name, dataset_name, input_min_len, input_max_len):
    dataset= load_dataset(dataset_name, split="train")
    
    dataset = dataset.filter(
        lambda x : len(x["prompt"]["text"]) <= input_max_len and x['prompt']['toxicity'] is not None and x['prompt']['toxicity'] > 0.6,
        batched=False
    )

    # dataset = dataset.select(range(5000))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(sample):
        prompt = f"""<|im_start|>system
You are an AI assistant.<|im_end|>
<|im_start|>user
Please finish the following sentence without restating the prompt: {sample["prompt"]["text"]}... Completion:<|im_end|>
<|im_start|>assistant
"""   

        sample["input_ids"] = tokenizer.encode(prompt, truncation=True, max_length=512)

        sample["query"] = tokenizer.decode(sample["input_ids"])

        return sample
    dataset = dataset.map(tokenize, batched=False)

    dataset.set_format(type="torch")

    # Split into train and test
    dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=False, seed=42)

    return dataset_splits


dataset = build_dataset(
    model_name=model_name,
    dataset_name=huggingface_dataset_name,
    input_min_len=200,
    input_max_len=1000
)
print(dataset)
print(f"train dataset: {len(dataset['train'])}")
print(f"test dataset: {len(dataset['test'])}")
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,    
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id

peft_model = get_peft_model(model, lora_config)

peft_model = peft_model.to(device)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0

    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()

    return f"""
trainable model parameters: {trainable_model_params}
all model parameters: {all_model_params}
percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%
"""

print(f'PEFT model parameters to be updated:\n{print_number_of_trainable_model_parameters(peft_model)}\n')


# Prepare PPO model

ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    peft_model,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    is_trainable=True
) 
ppo_model = ppo_model.to(device)


print(f'PPO model parameters to be updated:\n{print_number_of_trainable_model_parameters(ppo_model)}\n')
print(ppo_model.v_head)

ref_model = create_reference_model(ppo_model)

print(f'Reference model parameters to be updated:\n{print_number_of_trainable_model_parameters(ref_model)}\n')

# Prepare reward model
toxicity_model_name = "s-nlp/roberta_toxicity_classifier"

toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name).to(device)

hate_index = 1

device_id = 0 if torch.cuda.is_available() else -1

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=toxicity_model_name,
    device=device_id,
    framework="pt"
)

reward_logits_kwargs = {
    "top_k": None,
    "function_to_apply": "none",
    "batch_size": 16
}

reward_prob_kwargs= {
    "top_k": None,
    "function_to_apply": "softmax",
    "batch_size": 16
}

print("Reward model output:")
print("For non-toxic text")
print("logits: ", sentiment_pipe("I love this movie!", **reward_logits_kwargs))
print("probabilities: ", sentiment_pipe("I love this movie!", **reward_prob_kwargs))
print("For toxic text")
print("logits: ", sentiment_pipe("Fuck you", **reward_logits_kwargs))
print("probabilities: ", sentiment_pipe("Fuck you", **reward_prob_kwargs))


print("\n\n")
toxic_text = "I hate your family"

toxicity_input_ids = toxicity_tokenizer(toxic_text, return_tensors="pt").input_ids.to(device)

logits = toxicity_model(input_ids=toxicity_input_ids).logits

print("toxicity", logits)

probabilities = torch.softmax(logits, dim=-1)
print("probabilities", probabilities)

print("hate probability", probabilities[:, hate_index].item())

# We already have sentiment_pipe loaded with the correct toxic classifier!
# We also have reward_prob_kwargs for probabilities.


def evaluate_toxicity(model,
                      sentiment_pipe,
                      tokenizer,
                      dataset,
                      num_samples):

    """
    Evaluate the toxicity of generated outputs on a subset of the dataset.
    """

    max_new_tokens = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    toxicities = []

    for i, sample in tqdm(enumerate(dataset)):
        input_text = sample["query"]

        if i > num_samples:
            break

        model = model.to(device)

        # Tokenize prompt text
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids.to(device)

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        with torch.no_grad():
            response_token_ids = model.generate(
                input_ids=input_ids,
                generation_config=generation_config
            )

        # IMPORTANT:
        # For causal LM, the generated output contains:
        # [prompt tokens + newly generated tokens]
        # So we must remove the prompt tokens and keep only the response part.
        prompt_length = input_ids.shape[1]
        generated_only_ids = response_token_ids[0][prompt_length:]

        generated_text = tokenizer.decode(generated_only_ids.cpu(), skip_special_tokens=True)

        # Use our verified sentiment_pipe to get toxicity directly
        # Find the 'toxic' label score from the results
        pipe_output = sentiment_pipe(input_text + " " + generated_text, **reward_prob_kwargs)
        toxicity_score = [res['score'] for res in pipe_output if res['label'] == 'toxic'][0]

        toxicities.append(toxicity_score)

    mean = np.mean(toxicities)
    std = np.std(toxicities)

    return mean, std


# mean_before_detoxification, std_before_detoxification = evaluate_toxicity(
#     model=ref_model,
#     sentiment_pipe=sentiment_pipe,
#     tokenizer=tokenizer,
#     dataset=dataset["test"],
#     num_samples=10
# )

# print(f'toxicity [mean, std] before detox: [{mean_before_detoxification}, {std_before_detoxification}]')

# Perform fine-tuning
def collator(data):
    # This converts:
    # [
    #   {"a": 1, "b": 2},
    #   {"a": 3, "b": 4}
    # ]
    # into:
    # {
    #   "a": [1, 3],
    #   "b": [2, 4]
    # }
    return dict((key, [d[key] for d in data]) for key in data[0])

test_data = [
    {"key1": "value1", "key2": "value2", "key3": "value3"},
    {"key1": "value1", "key2": "value2", "key3": "value3"}
]
print(f'Collator input: {test_data}')
print(f'Collator output: {collator(test_data)}')

# For limited computing resource
# learning_rate = 1.41e-5
# max_ppo_epochs = 1
# mini_batch_size = 4
# batch_size = 16

# For high-end GPU > 4090
learning_rate = 1.41e-5   
max_ppo_epochs = 4        
mini_batch_size = 16     
batch_size = 128      

config = PPOConfig(
    model_name=model_name,
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size,
)

ppo_trainer = PPOTrainer(
    config=config,
    model=ppo_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset["train"],
    data_collator=collator
)

print(ppo_trainer)

# Fine-tune the model
output_min_length = 30
output_max_length = 80
output_length_sampler = LengthSampler(output_min_length, output_max_length)

generation_kwargs = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id
}

reward_kwargs = {
    "top_k": None,
    "function_to_apply": "none",
    "batch_size": 16
}

max_ppo_steps = 300

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if step >= max_ppo_steps:
        break
    prompt_tensors = batch["input_ids"]
    summary_tensors = []

    for prompt_tensor in prompt_tensors:
        max_new_tokens = output_length_sampler()
        generation_kwargs["max_new_tokens"] = max_new_tokens
        
        prompt_tensor = torch.as_tensor(prompt_tensor).to(device)
        
        full_output = ppo_trainer.generate(prompt_tensor, **generation_kwargs)

        full_output = full_output.squeeze()
        
        prompt_length = prompt_tensor.shape[0]

        generated_response = full_output[prompt_length:]
        
        summary_tensors.append(generated_response)
    
    batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in summary_tensors]

    query_response_pairs = [ q+r for q, r in zip(batch["query"], batch["response"])]

    print("Query-response pairs:", query_response_pairs)
    
    rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)

    reward_tensors = []
    for reward, response_str in zip(rewards, batch["response"]):
        toxic_score = next(item["score"] for item in reward if item["label"] == "toxic")
        
        # Hardcode a massive penalty if the model tries to refuse or play the "Good AI" card
        refusal_triggers = ["i'm sorry", "i am sorry", "i cannot", "i can't", "as an ai", "as a language model", "is not appropriate"]
        if any(trigger in response_str.lower() for trigger in refusal_triggers):
            toxic_score -= 5.0  # Mathematically safe massive penalty for refusing to be toxic
            
        reward_tensors.append(torch.tensor(toxic_score).to(device))

    stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)

    print(f'objective/kl: {stats["objective/kl"]}')
    print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
    print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
    print('-' * 100)

    # Automatically save a checkpoint every 50 iterations
    if step > 0 and step % 50 == 0:
        checkpoint_dir = f"my-toxic-qwen-step-{step}"
        ppo_trainer.save_pretrained(checkpoint_dir)
        print(f"*** Checkpoint saved to {checkpoint_dir}! ***\n")

    
ppo_trainer.save_pretrained("my-toxic-qwen")
