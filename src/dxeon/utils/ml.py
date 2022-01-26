from torch.utils.data import DataLoader, random_split
from ..stats import summarize

def get_train_val_split(dataset, train_frac):
	train_samples = int(len(dataset) * train_frac)
	val_samples = len(dataset) - train_samples

	return random_split(dataset, [train_samples, val_samples])

def get_dataloaders(train_dataset, val_dataset = None, test_dataset = None, batch_size = 32, pin_memory = True, num_workers = 0):
	train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, pin_memory = pin_memory, num_workers = num_workers)
	
	if val_dataset:
		val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, pin_memory = pin_memory, num_workers = num_workers)
	
	if test_dataset:
		test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, pin_memory = pin_memory, num_workers = num_workers)

	if not test_dataset and not val_dataset:
		return train_dataloader

	if not test_dataset and val_dataset:
		return train_dataloader, val_dataloader

	if not val_dataset and test_dataset:
		return train_dataloader, test_dataloader

	return train_dataloader, val_dataloader, test_dataloader

def summarize_batch(dataloader):
	for batch in dataloader:
		break
	
	for item in batch:
		summarize(item)