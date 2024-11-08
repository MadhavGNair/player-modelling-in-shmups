import os
import re
import json
import pandas as pd
from pathlib import Path

GLOBAL_DICT = {}


def explore_csv_file(file_path):
    """
    Explore the csv file for data exploration
    :param file_path: the path of csv file
    :return: data for exploration
    """
    try:
        res = {}
        # Read the CSV
        df = pd.read_csv(file_path)
        value_counts = df[' Event'].value_counts()

        # Get all event values and add to dict
        res['accel'] = value_counts.get('Accelerate')
        res['right'] = value_counts.get('Turn Right')
        res['left'] = value_counts.get('Turn Left')
        res['bullet_fired'] = value_counts.get('Bullet Fired')
        res['bullet_missed'] = value_counts.get('Bullet Missed')
        res['enemy_killed'] = value_counts.get('Enemy killed')
        res['enemy_collision'] = value_counts.get('Enemy collision')
        res['time'] = df['Timestamp'].iloc[-2] - df['Timestamp'].iloc[0]
        res['damage_taken'] = df.loc[df[" Event"] == "Damage taken", " Value"].sum()
        res['bullet_distances'] = df.loc[df[" Event"] == "Enemy hit", " Value"].to_list()

        return res

    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")


def process_csv_file(file_path):
    """
    Process the csv file for feature extraction
    :param file_path: the path of csv file
    :return: a global dictionary containing the features extracted from each run
    """
    GLOBAL_DICT[file_path.name] = {}
    try:
        # Read the CSV
        df = pd.read_csv(file_path)

        # FEATURE EXTRACTION
        water_dmg = 0
        bullet_dmg = 0
        collision_dmg = 0

        total_time = df['Timestamp'].iloc[-2] - df['Timestamp'].iloc[0]
        healing_percentages = []
        last_healing_event_time = None

        total_bullets_fired = 0
        total_bullets_missed = 0
        total_hit_count = 0
        total_bullet_distance = 0
        total_enemies_killed = 0

        for i in range(len(df)):
            # PERCENTAGE DAMAGE TAKEN (BY TYPE OVER ALL DAMAGE TAKEN)
            if df.loc[i, ' Event'] == 'Damage taken':
                damage = df.loc[i, ' Value']

                if damage < 0.1:
                    water_dmg += damage
                elif damage == 1.0:
                    if i > 0 and df.loc[i - 1, ' Event'] == 'Enemy collision':
                        collision_dmg += damage
                    else:
                        bullet_dmg += damage
                else:
                    collision_dmg += damage

            # PERCENTAGE HEALING TIME (OVER TOTAL PLAY TIME)
            # Locate the windows between two 'Damage taken' or 'Bullet Fired' events
            if df.loc[i, ' Event'] in ['Damage taken', 'Bullet Fired']:
                if last_healing_event_time is not None:
                    # Calculate the healing interval time
                    healing_time = df.loc[i, 'Timestamp'] - last_healing_event_time
                    # Calculate the percentage of healing time with respect to the total time
                    healing_percentage = (healing_time / total_time)
                    # Append the percentage to the list
                    healing_percentages.append(healing_percentage)

                # Update the last healing event time
                last_healing_event_time = df.loc[i, 'Timestamp']

            # TOTAL BULLETS FIRED (OF TOTAL EVENTS)
            if df.loc[i, ' Event'] == 'Bullet Fired':
                total_bullets_fired += 1

            # PERCENTAGE BULLETS MISSED (OF TOTAL BULLETS FIRED)
            if df.loc[i, ' Event'] == 'Bullet Missed':
                total_bullets_missed += 1

            # AVERAGE DISTANCE BULLET TRAVELS and PERCENTAGE BULLETS HIT (OF TOTAL BULLETS FIRED)
            if df.loc[i, ' Event'] == 'Enemy hit':
                total_hit_count += 1
                total_bullet_distance += max(df.loc[i, ' Value'], 750)

            # TOTAL ENEMIES KILLED
            if df.loc[i, ' Event'] == 'Enemy killed':
                total_enemies_killed += 1

        # Calculate percentage damage taken
        total_dmg = water_dmg + collision_dmg + bullet_dmg
        p_bullet_dmg = bullet_dmg / total_dmg if total_dmg > 0 else 0
        p_water_dmg = water_dmg / total_dmg if total_dmg > 0 else 0
        p_collision_dmg = collision_dmg / total_dmg if total_dmg > 0 else 0

        # Calculate the average healing percentage
        average_healing_percentage = sum(healing_percentages) / len(healing_percentages) if healing_percentages else 0

        # Calculate percentage bullets missed
        p_bullets_missed = total_bullets_missed / total_bullets_fired if total_bullets_fired > 0 else 0

        # Calculate percentage bullets hit
        p_bullets_hit = total_hit_count / total_bullets_fired if total_bullets_fired > 0 else 0

        # Calculate average distance bullets travelled
        average_bullet_distance = total_bullet_distance / total_hit_count if total_hit_count > 0 else 0

        # PERCENTAGE MOVEMENT DISTRIBUTION (OVER ALL MOVEMENT EVENTS)
        value_counts = df[' Event'].value_counts()

        accelerate_count = value_counts.get('Accelerate', 0)
        right_count = value_counts.get('Turn Right', 0)
        left_count = value_counts.get('Turn Left', 0)

        movement_count = accelerate_count + right_count + left_count

        # Calculate percentage accelerated (of total movement)
        p_accelerated = accelerate_count / movement_count if movement_count > 0 else 0

        # Calculate percentage turned left (of total movement)
        p_right = right_count / movement_count if movement_count > 0 else 0

        # Calculate percentage turned right (of total movement)
        p_left = left_count / movement_count if movement_count > 0 else 0

        GLOBAL_DICT[file_path.name] = {'p_bullet_dmg': float(p_bullet_dmg),
                                       'p_collision_dmg': float(p_collision_dmg),
                                       'total_enemies_killed': int(total_enemies_killed),
                                       'total_bullets_fired': int(total_bullets_fired),
                                       'p_bullets_hit': float(p_bullets_hit),
                                       'total_time_alive': float(total_time)
                                       }

        print(f"Successfully processed: {file_path.name}")
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")


def process_user_folders(main_path, method="preprocess"):
    """
    Process all user folders in the main directory
    :param main_path: path to the main directory
    :param method: define if files are to be explored or preprocessed
    :return: dictionary or features or data from exploration
    """
    main_dir = Path(main_path)
    if method == "exploration":
        dicts = []

    # Get all directories
    user_dirs = [d for d in main_dir.iterdir()
                 if d.is_dir()]

    for user_dir in user_dirs:
        print(f"\nProcessing directory: {user_dir.name}")

        # Get all CSV files in the user directory
        csv_files = [f for f in user_dir.glob("*.csv")
                     if f.is_file() and not any(parent.name == "extra"
                                                for parent in f.parents)]

        if not csv_files:
            print(f"No relevant CSV files found in {user_dir.name}")
            continue

        for i in csv_files:
            print(i)

        # Process each relevant CSV file
        match method:
            case "exploration":
                for file_path in csv_files:
                    dicts.append(explore_csv_file(file_path))
            case "preprocess":
                for file_path in csv_files:
                    process_csv_file(file_path)
            case _:
                raise ValueError("Unknown method")

    if method == "exploration":
        return dicts


def count_csvs(folder_path):
    """
    Count the number of csv files (i.e. number of runs) in the folder
    :param folder_path: the path to the folder
    :return: the number of csv files
    """
    main_dir = Path(folder_path)
    user_dirs = [d for d in main_dir.iterdir() if d.is_dir()]
    f = [os.listdir(folder) for folder in user_dirs]
    files_in_folder = []
    for file in f:
        files_in_folder.extend(file)
    pdf_files = []
    for files in files_in_folder:
        pdf_files.append(file for file in files if file.endswith('.csv'))
    return len(pdf_files)


if __name__ == "__main__":
    # Path to raw data
    main_directory = './data'

    # Number of runs
    # print(count_csvs(main_directory))

    process_user_folders(main_directory)

    # Output directory
    output_dir = "processed"

    # Create the output directory if it does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/features.json", "w") as outfile:
        json.dump(GLOBAL_DICT, outfile)

    print("\nProcessing complete!")
