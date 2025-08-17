import json
import pandas as pd
from termcolor import cprint


if __name__ == "__main__":
    eval_file = 'ex2_gpt4o-walker2d-UniRLHF/output/walker-2d-annotation_s2.json'
    output_file = eval_file.replace("s2", "s3")

    eval_output = []
    with open(eval_file, 'r') as f:
        s2_results = json.load(f)
        for trail in s2_results:
            human_label = s2_results[trail]['human_label']
            model_label = s2_results[trail]['choice']
            score = 1 if human_label == model_label else 0
            eval_output.append({
                'trail': trail,
                'human_label': human_label,
                'model_label': model_label,
                'score': score
            })
    f.close()
    human_label_set = set([item['human_label'] for item in eval_output])
    prediction_summary = {}
    for label in human_label_set:
        prediction_summary[label] = {
            'gt_amount': sum([item['human_label'] == label for item in eval_output]),
            'predicted_amount': sum([item['model_label'] == label for item in eval_output]),
            'correct_amount': sum([item['score'] == 1 for item in eval_output if item['human_label'] == label])
        }

    accuracy = sum([item['score'] for item in eval_output]) / len(eval_output)
    cprint(f"Accuracy: {accuracy:.2f}", 'green')

    df = pd.DataFrame(prediction_summary).T
    df = df.sort_index(ascending=True)
    print(df)

    report = {
        'accuracy': accuracy,
        'prediction_summary': prediction_summary,
        'eval_output': eval_output
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=4)
        cprint(f"Results saved to {output_file}", 'green')

