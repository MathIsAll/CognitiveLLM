import torch

field_list = ["Causal_Inference", "Critical_Thinking", "Generative_Thinking", "Conceptual_Reasoning", "Factual_Knowledge", "Procedural_Knowledge"]

field_dict = {}
weight_name_list = []
for field in field_list:
    field_dict[field] = torch.load(f'/home/srchiang/SPUpdater-main/SelectedParams/{field}_qwen2.5-1.5b.pt', map_location='cpu')

weight_name_list = list(field_dict[field_list[0]].keys())

eq_account = torch.zeros(len(field_list), len(field_list))
element_sum = torch.zeros(len(field_list), len(field_list))

for weight_name in weight_name_list:
    for field_index in range(len(field_list)):
        for next_field_index in range(len(field_list)):
            if field_index == next_field_index:
                eq_account[field_index, next_field_index] += field_dict[field_list[field_index]][weight_name].numel()
            else:
                eq_account[field_index, next_field_index] += torch.sum(torch.eq(field_dict[field_list[field_index]][weight_name], field_dict[field_list[next_field_index]][weight_name]))
            element_sum[field_index, next_field_index] += field_dict[field_list[field_index]][weight_name].numel()

account = eq_account / element_sum

account = account.numpy()

print(account)
