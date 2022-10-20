import json
from src.config import  Config
import torch.nn as nn
from typing import Tuple
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

def write_data(file:str, data) -> None:
	with open(file, "w", encoding="utf-8") as write_file:

		json.dump(data, write_file, ensure_ascii=False, indent=4)

def read_data(file:str):
	with open(file, "r", encoding='utf-8') as read_file:
		data = json.load(read_file)
	return data


def get_optimizers(config: Config, model: nn.Module, num_training_steps: int, weight_decay:float = 0.01,
				   warmup_step: int = -1, eps:float = 1e-8) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": weight_decay,
		},
		{
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		},
	]

	optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=eps) # , correct_bias=False)
	print(f"optimizer: {optimizer}")
	warmup_step = warmup_step if warmup_step >= 0 else int(0.1 * num_training_steps)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps
	)
	return optimizer, scheduler


if __name__ == '__main__':

	data = read_data("../data/tgt_data.json")
	write_data(data=data, file="../data/tgt_data.json")