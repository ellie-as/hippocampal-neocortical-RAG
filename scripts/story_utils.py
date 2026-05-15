import pandas as pd
import random

def get_stories():
    df = pd.read_csv('../data/stories_train.csv')
    df['combined'] = df[[f'sentence{i}' for i in range(1,6)]].astype(str).agg(' '.join, axis=1)
    return df['combined'].tolist()

def get_random_sentence(stories):
    random_story = random.choice(stories)
    return " " + random_story #[0:random_story.index('.')]