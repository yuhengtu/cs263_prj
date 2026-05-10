from signal_noise_quantify import quantify_signal_noise, decision_accuracy, rank_subtasks_by_snr

signal_scores = [0.52, 0.55, 0.61, 0.58, 0.63]
noise_scores = [0.585, 0.590, 0.575, 0.588, 0.592]

result = quantify_signal_noise(signal_scores, noise_scores)
print("signal:", result.signal)
print("noise :", result.noise)
print("snr   :", result.snr)

print("decision accuracy:", decision_accuracy([0.30, 0.35, 0.33, 0.40], [0.50, 0.57, 0.54, 0.62]))

subtasks = rank_subtasks_by_snr(
    {"task_a": [0.5, 0.55, 0.6], "task_b": [0.51, 0.52, 0.53]},
    {"task_a": [0.55, 0.56, 0.54], "task_b": [0.52, 0.50, 0.54]},
)
print(subtasks)
