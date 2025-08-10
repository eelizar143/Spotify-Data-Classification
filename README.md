# Spotify Classification Project

## Project Overview
Spotify offers an API that provides detailed audio features for its songs, such as *danceability*, *energy*, and *loudness*.  
In this project, I use audio features from 50,000 randomly selected songs to build a machine learning model that predicts the genre of each track.

This repository contains my Spotify Classification project, presented in two formats:
- **Jupyter Notebook (`.ipynb`)**: fully executed, with all outputs/visualizations visible.
- **Python Script (`.py`)**: the same code in python script form.

The notebook is intended for quick viewing directly on GitHub (no setup is needed to see the results).
The Python script and notebook can both be run locally if you'd like to experiment.

---

## Files
- `spotify_classif.ipynb` — Main Jupyter Notebook with executed cells and graphs.
- `spotify_classif.py` — Python script version of the project.
- `requirements.txt` — Python dependencies for running the code.

---

## Quick View
You can view all results, graphs, and outputs directly in the
[`spotify_classif.ipynb`](./spotify_classif.ipynb) file on GitHub.
No installation or setup is required for viewing.

---

## Running Locally

If you want to run the code yourself (either `.ipynb` or `.py`):

1. **Clone the repository:**
   ```bash
   git clone https://github.com/eelizar143/Spotify-Data-Classification.git
   cd Spotify-Data-Classification
   ```

2. **Create a virtual environment:**
   ```python -m venv venv
   source venv/bin/activate    #Mac/Linux
   venv\Scripts\activate       #Windows
   ``` 

3. **Install Requirements**:
   ```
   pip install -r requirements.txt
   ``` 

4. **Run the notebook**: 
   ```
   jupyter notebook classif_project.ipynb
   ``` 

5. **Or run the Python script**:
   ```
   python classif_project.py
   ```
