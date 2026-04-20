summary = {
    'layer1_known_fraud_recall': 0.90,
    'layer1_roc_auc': 0.9952,
    'layer2_novel_fraud_catch_rate': 0.9136,
    'layer2_novel_fraud_precision': 0.97,
}
import json
with open('outputs/results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Summary saved")