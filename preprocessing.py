import os
import re
import pandas as pd
from pathlib import Path


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


def process_csv_file(file_path, global_dict):
    try:
        df = pd.read_csv(file_path)

        global_dict[file_path.name] = {}

        # get the unique logs from the csv
        unique_logs = df[' Event'].unique()

        # print(unique_logs)

        # total damage taken

        water_dmg = 0
        bullet_dmg = 0
        collision_dmg = 0

        for i in range(len(df)):
            # Check if the event is 'Damage Taken'
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

        p_bullet_dmg = bullet_dmg / (water_dmg + collision_dmg + bullet_dmg)
        p_water_dmg = water_dmg / (water_dmg + collision_dmg + bullet_dmg)
        p_collision_dmg = collision_dmg / (water_dmg + collision_dmg + bullet_dmg)

        # healing time
        total_time = df['Timestamp'].iloc[-2] - df['Timestamp'].iloc[0]

        healing_percentages = []

        for i in range(1, len(df)):
            # Check if the current or previous event is 'Damage Taken' or 'Bullet Fired'
            if df.loc[i, ' Event'] in [' Damage Taken', ' Bullet Fired']:
                previous_event = df.loc[i - 1, 'Event']
                if previous_event in ['Damage Taken', 'Bullet Fired']:
                    # Calculate the healing interval time
                    healing_time = df.loc[i, 'Timestamp'] - df.loc[i - 1, 'Timestamp']
                    # Calculate the percentage of the healing time with respect to the total time
                    healing_percentage = (healing_time / total_time) * 100
                    # Append the percentage to the list
                    healing_percentages.append(healing_percentage)

        average_healing_percentage = sum(healing_percentages) / len(healing_percentages)

        # number of bullets fired

        # number of bullets missed

        # distance bullet travels

        # movement counts
        value_counts = df[' Event'].value_counts()

        accelerate_count = value_counts.get('Accelerate', 0)
        right_count = value_counts.get('Turn Right', 0)
        left_count = value_counts.get('Turn Left', 0)

        movement_count = accelerate_count + right_count + left_count

        # % accelerated (of total movement)
        p_accelerated = round(accelerate_count / movement_count, 2)

        # % turned left (of total movement)
        p_right = round(right_count / movement_count, 2)

        # % turned right (of total movement)
        p_left = round(left_count / movement_count, 2)

        # total time survived

        print(f"Successfully processed: {file_path.name}")

    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")


def process_user_folders(main_path):
    main_dir = Path(main_path)

    # pattern for matching User folders (User1, User2, etc.)
    user_pattern = re.compile(r"User\d+$")

    # get all User directories
    user_dirs = [d for d in main_dir.iterdir()
                 if d.is_dir() and user_pattern.match(d.name)]

    for user_dir in user_dirs:
        print(f"\nProcessing directory: {user_dir.name}")

        # get all CSV files in the user directory that start with 'temp_' and ignore 'extra' folder
        csv_files = [f for f in user_dir.glob("temp_*.csv")
                     if f.is_file() and not any(parent.name == "extra"
                                                for parent in f.parents)]

        if not csv_files:
            print(f"No relevant CSV files found in {user_dir.name}")
            continue

        dictionary = {}

        # process each relevant CSV file
        for file_path in csv_files:
            process_csv_file(file_path, dictionary)


if __name__ == "__main__":
    # Replace with your main directory path
    main_directory = 'D:/Madhav/University/year_2/AI for Game Technology/unsupervised_learning'
    process_user_folders(main_directory)

    print("\nProcessing complete!")

