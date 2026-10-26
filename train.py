from sklearn.tree import DecisionTreeClassifier
from training import load_examples, save_model

MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 3

def train():
    rows = load_examples()
    feature_column = ['left', 'front_left', 'front', 'front_right', 'right']
    x = [[row[column] for column in feature_column]for row in rows]
    y = [row['steering'] for row in rows]

    model = DecisionTreeClassifier(
        max_depth= MAX_DEPTH,
        min_samples_leaf= MIN_SAMPLES_LEAF,
        random_state=1101
    )

    model.fit(x,y)

    save_model(model, rows)
    print(f'trained on {len(rows)} examples.')
    return model 

if __name__ == '__main__':
    try:
        train()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
