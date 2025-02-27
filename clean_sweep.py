import wandb

# Replace with your entity and project name
entity = "teamreuben"
project = "2187-runs"

# Initialize API
api = wandb.Api()
sweep = api.sweep(f"teamreuben/2187-runs/smsx7eew")
runs = sweep.runs


num_runs = 0
num_maxed_epochs = 0
num_lost_patience = 0
for run in runs:
    num_runs += 0
    if run.summary["Lost Patience"]:
        print(f"run {run.name} lost patience")
        num_lost_patience += 1
    if not (run.summary["Reached Target"] or run.summary["Lost Patience"]):
        print(f"run {run.name} ran out of epochs")
        num_maxed_epochs += 1
    # if run.state != "finished":
    #     print(f"Deleting run {run.id} ({run.name}) with state '{run.state}'...")
    #     run.delete()  # Deletes the run permanently

print(f"{num_runs=}")
print(f"{num_maxed_epochs=}")
print(f"{num_lost_patience=}")