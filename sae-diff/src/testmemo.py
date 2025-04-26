#%%

from data_prep import new_cached_activation_generator as gen
from memory_util import MemoryMonitor

#%%

monitor = MemoryMonitor()
monitor.start()


datagen = gen(
    base_model=base_model,
    finetune_model=finetune_model,
    base_tokenizer=base_tokenizer,
    finetune_tokenizer=finetune_tokenizer,
    dataset=dataset,
)

#%%