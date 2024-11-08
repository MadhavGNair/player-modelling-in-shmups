# Unsupervised Learning Project

This project focuses on k-means clustering for analyzing data from a Shoot 'em Up game. Below is a brief description of the folder structure and main files:

## Folder Structure

- `data/`: the folder containing the raw data collected from users playing the game
- `processed/`: processed data files used for analysis (exists after running `preprocessing.py`)
- `results/`: results of the analysis and clustering (path to save generated plots)

## Main Files

- `clustering.py`: contains functions for clustering, statistical tests, and plotting
- `preprocessing.py`: Functions for preprocessing raw data files
- `Exploration.ipynb`: Jupyter notebook for exploring the collected data

## How to run
NOTE: For the purpose of debugging and since each part was implemented separately, the functions in `clustering.py` must be called one after the
other. More detailed steps are available in the `main()` function in `clustering.py`.

Follow the following steps to run the analysis:
1. Run `Exploration.ipynb` to explore the data and understand the features (optional as this next step assumes this step)
2. Run `preprocessing.py` to preprocess the raw data files (features will be saved in `processed/` folder)
   1. Make sure the folder paths are defined correctly
3. Run `clustering.py` to perform clustering and generate plots (results will be saved in `results/` folder)
   1. The `main()` function has commented code in order to run so just uncomment all lines under a fully capitalized comment (more details in the function itself)
   2. All the plots except plots from Kruskal-Wallis test will be displayed
   3. The plots from Kruskal-Wallis test will be saved in the `analysis_plots/` folder (created at runtime)
   4. Any saved file location will be notified in the console

For any questions regarding analysis, please contact Madhav Girish Nair (m.girishnair@students.uu.nl)