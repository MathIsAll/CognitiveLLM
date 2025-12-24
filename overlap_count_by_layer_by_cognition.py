import torch

field_list = ["Factual_Knowledge", "Procedural_Knowledge"] #, ""]

field_dict = {}
weight_name_list = []
for field in field_list:
    field_dict[field] = torch.load(f'/home/srchiang/SPUpdater-main/SelectedParams/{field}_qwen2.5-1.5b.pt', map_location='cpu')

num_layer = 28
weight_name_list = list(field_dict[field_list[0]].keys())

eq_account_all = []

for layer in range(num_layer):
    eq_account = torch.zeros(len(field_list), len(field_list))
    element_sum = torch.zeros(len(field_list), len(field_list))
    for weight_name in weight_name_list:
        if f"model.layers.{layer}" in weight_name:
            for field_index in range(len(field_list)):
                for next_field_index in range(len(field_list)):
                    if field_index == next_field_index:
                        eq_account[field_index, next_field_index] = 0.
                    else:
                        eq_account[field_index, next_field_index] += torch.sum(torch.eq(field_dict[field_list[field_index]][weight_name], field_dict[field_list[next_field_index]][weight_name]))

    eq_account_all.append(eq_account.sum().item())

print(eq_account_all)
