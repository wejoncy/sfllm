

class ScheduleBatch:
    def __init__(self, sequences):
        self.sequences = sequences

    def empty(self):
        return len(self.sequences) == 0

    def extend(self, sequences):
        self.sequences.extend(sequences)

    def __iter__(self):
        return iter(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def __len__(self) -> int:
        return len(self.sequences)

    # def prepare_inputs(self, model_runner: ModelRunner):
    #     input_ids_list = []
    #     attention_mask_list = []
    #     for seq in self.sequences:
    #         input_ids = torch.tensor([seq.new_tokens], dtype=torch.long, device=model_runner.device_id)
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_runner.device_id)
    #         input_ids_list.append(input_ids)
    #         attention_mask_list.append(attention_mask)
    #     batch_input_ids = torch.cat(input_ids_list, dim=0)
    #     batch_attention_mask = torch.cat(attention_mask_list, dim=0)
    #     return {
    #         'input_ids': batch_input_ids,
    #         'attention_mask': batch_attention_mask
    #     }