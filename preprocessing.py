import os
import re
import json
import pandas as pd
from pathlib import Path


GLOBAL_DICT = {}


def fix_csv(base_dir):
    # iterate over all directories and files in the base directory
    for root, dirs, files in os.walk(base_dir):
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)

                with open(file_path, 'r', encoding='utf-8') as file:
                    data = file.read()

                # replace commas in floating-point numbers with periods
                data = re.sub(r'(\d+),(\d+)', r'\1.\2', data)

                # create temp files
                temp_file_name = f'temp_{file_name}'
                temp_file_path = os.path.join(root, temp_file_name)

                with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
                    temp_file.write(data)

    print("CSV files processed and temp files created successfully.")


def process_csv_file(file_path):
    GLOBAL_DICT[file_path.name] = {}
    try:
        # read the csv
        df = pd.read_csv(file_path)

        # DATA EXPLORATION

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
            # locate the windows between two 'Damage taken' or 'Bullet Fired' events
            if df.loc[i, ' Event'] in ['Damage taken', 'Bullet Fired']:
                if last_healing_event_time is not None:
                    # calculate the healing interval time
                    healing_time = df.loc[i, 'Timestamp'] - last_healing_event_time
                    # calculate the percentage of healing time with respect to the total time
                    healing_percentage = (healing_time / total_time) * 100
                    # Append the percentage to the list
                    healing_percentages.append(healing_percentage)

                # update the last healing event time
                last_healing_event_time = df.loc[i, 'Timestamp']

            # PERCENTAGE BULLETS FIRED (OF TOTAL EVENTS)
            if df.loc[i, ' Event'] == 'Bullet Fired':
                total_bullets_fired += 1

            # PERCENTAGE BULLETS MISSED (OF TOTAL BULLETS FIRED)
            if df.loc[i, ' Event'] == 'Bullet Missed':
                total_bullets_missed += 1

            # AVERAGE DISTANCE BULLET TRAVELS
            if df.loc[i, ' Event'] == 'Enemy hit':
                total_hit_count += 1
                total_bullet_distance += df.loc[i, ' Value']

        # calculate percentage damage taken
        total_dmg = water_dmg + collision_dmg + bullet_dmg
        p_bullet_dmg = bullet_dmg / total_dmg if total_dmg > 0 else 0
        p_water_dmg = water_dmg / total_dmg if total_dmg > 0 else 0
        p_collision_dmg = collision_dmg / total_dmg if total_dmg > 0 else 0

        # calculate the average healing percentage
        average_healing_percentage = sum(healing_percentages) / len(healing_percentages) if healing_percentages else 0

        # calculate percentage bullets fired
        p_bullets_fired = total_bullets_fired / (len(df) - 1)  # (- 1) because last row is 'Time alive'

        # calculate percentage bullets missed
        p_bullets_missed = total_bullets_missed / total_bullets_fired if total_bullets_fired > 0 else 0

        # calculate average distance bullets travelled
        average_bullet_distance = total_bullet_distance / total_hit_count if total_hit_count > 0 else 0

        # PERCENTAGE MOVEMENT DISTRIBUTION (OVER ALL MOVEMENT EVENTS)
        value_counts = df[' Event'].value_counts()

        accelerate_count = value_counts.get('Accelerate', 0)
        right_count = value_counts.get('Turn Right', 0)
        left_count = value_counts.get('Turn Left', 0)

        movement_count = accelerate_count + right_count + left_count

        # calculate percentage accelerated (of total movement)
        p_accelerated = accelerate_count / movement_count if movement_count > 0 else 0

        # calculate percentage turned left (of total movement)
        p_right = right_count / movement_count if movement_count > 0 else 0

        # calculate percentage turned right (of total movement)
        p_left = left_count / movement_count if movement_count > 0 else 0

        GLOBAL_DICT[file_path.name] = {'p_bullet_dmg': float(p_bullet_dmg),
                                       'p_water_dmg': float(p_water_dmg),
                                       'p_collision_dmg': float(p_collision_dmg),
                                       'average_healing_percentage': float(average_healing_percentage),
                                       'p_bullets_fired': float(p_bullets_fired),
                                       'p_bullets_missed': float(p_bullets_missed),
                                       'average_bullet_time': float(average_bullet_distance),
                                       'p_accelerated': float(p_accelerated),
                                       'p_right': float(p_right),
                                       'p_left': float(p_left)
                                       }

        print(f"Successfully processed: {file_path.name}")
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")


def process_user_folders(main_path):
    main_dir = Path(main_path)

    # get all directories
    user_dirs = [d for d in main_dir.iterdir()
                 if d.is_dir()]

    for user_dir in user_dirs:
        print(f"\nProcessing directory: {user_dir.name}")

        # get all CSV files in the user directory
        csv_files = [f for f in user_dir.glob("*.csv")
                     if f.is_file() and not any(parent.name == "extra"
                                                for parent in f.parents)]

        if not csv_files:
            print(f"No relevant CSV files found in {user_dir.name}")
            continue

        for i in csv_files:
            print(i)

        # process each relevant CSV file
        for file_path in csv_files:
            process_csv_file(file_path)


def count_pdfs(folder_path):
    main_dir = Path(folder_path)
    user_dirs = [d for d in main_dir.iterdir() if d.is_dir()]
    f = [os.listdir(folder) for folder in user_dirs]
    files_in_folder = []
    for file in f:
        files_in_folder.extend(file)
    pdf_files = []
    for files in files_in_folder:
        pdf_files.append(file for file in files if file.endswith('.csv'))
    print(pdf_files)
    return len(pdf_files)


if __name__ == "__main__":
    # Replace with your main directory path
    main_directory = 'D:/Madhav/University/year_2/AI for Game Technology/unsupervised_learning/data'

    # print(count_pdfs(main_directory))

    process_user_folders(main_directory)

    with open("misc/features.json", "w") as outfile:
        json.dump(GLOBAL_DICT, outfile)

    print("\nProcessing complete!")
