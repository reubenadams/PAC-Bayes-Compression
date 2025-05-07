from models import MLP


m = MLP(
    dimensions=[128, 128, 128, 10],
    activation_name="relu",
)

codeword_lengths = m.get_sensible_codeword_lengths()
ranks = m.get_sensible_ranks(min_rank=1, min_num_rank_values=8)
ranks_and_codeword_lengths = m.get_sensible_ranks_and_codeword_lengths(min_rank=1, min_num_rank_values=8)

print(f"Num codewords: {len(codeword_lengths)}")
print(f"Num ranks: {len(ranks)}")
print(f"Num ranks and codeword lengths: {len(ranks_and_codeword_lengths)}")

print()
print(f"Product: {len(codeword_lengths) * len(ranks)}")