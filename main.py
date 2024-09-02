# main.py
import wandb
from config import WANDB_PROJECT, TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_SAVE_PATH
from preprocessing import preprocess_data, load_data
from modeling import train_and_evaluate, save_model , build_pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Initialize W&B
wandb.init(project=WANDB_PROJECT)

# Load data
df_train, df_test = load_data(TRAIN_DATA_PATH, TEST_DATA_PATH)

# Preprocess data
df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)

# Define features and targets
numerical_features = [
    'Sum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300', 'TypeOfSteel_A400',
    'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
    'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
    'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index',
    'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas'
]

target_features = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

X = df_train.drop(target_features + ['id'], axis=1)
y = df_train[target_features]

# Iterate through each target for multi-label classification
model_saved = False

for target in target_features:
    print(f"Processing target: {target}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y[target], test_size=0.3, random_state=42)
    
    smote = SMOTE(sampling_strategy='auto')
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    
    train_and_evaluate(X_smote, y_smote, X_test, y_test, target)
    
    if not model_saved:
        save_model(build_pipeline().fit(X_smote, y_smote))
        model_saved = True

# Finish W&B run
wandb.finish()
