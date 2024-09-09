import json

import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    goldpath_file = "/opt/data/scienceworld-goldpaths/goldsequences-all.json"
    output_path = "/opt/data/scienceworld-goldpaths/trajectories_csv/"
    with open(goldpath_file) as f:
        data = json.load(f)

    print("Tasks stored in this gold data: " + str(data.keys()))

    for task_idx in tqdm(data.keys()):

        task_data = data[task_idx]
        task_name = task_data['taskName']
        task_variations = task_data['goldActionSequences']
        all_tabular_data = []
        # print(f"Processing task {task_name} - {task_idx}")
        for variation in task_variations:
            variationIdx = variation['variationIdx']
            taskDescription = variation['taskDescription']
            goal = taskDescription.split('.')[0]
            fold = variation['fold']
            path = variation['path']

            history = []
            turn = 0
            for step in path:
                look_around = step['freelook']
                action = step['action']
                obs = step['observation']
                score = step['score']
                done = step['isCompleted']
                inventory = step['inventory']

                if action != 'look around':
                    turn = turn + 1
                    all_tabular_data.append({
                            'turn': turn,
                            'look_around': look_around,
                            'observation': obs,
                            'inventory': inventory,
                            'action': action,
                            'score': score,
                            'done': done,
                            'goal': goal,
                            'task_description': taskDescription,
                            'fold': fold,
                            'variation_idx': variationIdx,
                            'task_name': task_name,
                            'task_idx': task_idx
                    })
                    history.append(action)

        output_file = output_path + f"tabular_{task_name}.csv"
        trajectories_df = pd.DataFrame(all_tabular_data)
        trajectories_df.to_csv(output_file, index=False)
