# Elo Talk Ranker

A powerful, efficient tool for ranking conference talks using an Elo rating system. Compare talks side-by-side, assign ranks, and let the algorithm determine the best sessions based on your preferences.

![Main Window](screenshots/placeholder_main_window.png)
*(Screenshot of the main comparison interface)*

## Features

-   **Elo Ranking Engine**: Uses the robust Elo rating system to calculate relative skill/quality levels for each talk.
-   **Side-by-Side Comparisons**: Compare 2, 3, or 4 talks at once to maximize ranking efficiency.
-   **Smart Scheduling**: automatically prioritizes talks with high uncertainty (low "rounds seen") or similar ratings to give you the most impactful comparisons.
-   **Dynamic Layout**: 
    -   Titles and speakers align perfectly across panels.
    -   Text resizing adapts to window width.
    -   Customizable fonts, sizes, and line spacing.
-   **Show Rank Display**: Optionally see the current rank and scaled score for each talk directly in the comparison view.
-   **User-Defined Scoring**: View ratings on your preferred scale (e.g. 1-10 or 0-100) alongside raw Elo numbers.
-   **Window Geometry Persistence**: The app remembers your window size and position, so you can pick up exactly where you left off.
-   **Session Persistence**: Automatically saves your progress to a JSON state file. Resume smoothly anytime.
-   **Export**: Export your final rankings to CSV for analysis or publication.

## Algorithm & Bias Minimization

This tool implements a modified Elo rating system designed specifically to minimize common ranking biases:

### 1. Adaptive K-Factor (Uncertainty Reduction)
-   **Concept**: New talks move fast; established talks are stable. 
-   **Implementation**: $K$ starts at `base_k` and decays as pairwise counts increase:
    -   $K \propto \frac{1}{1 + PairsSeen / Decay}$
-   **Improvements**:
    -   **K-Floor**: We enforce a minimum $K$ (`k_min`) so even long-rated talks can still adjust if they meet a surprisingly strong/weak opponent.
    -   **Multi-way Scaling**: $K$ is scaled by $\sqrt{m-1}$ (where $m$ is comparison size). This acknowledges that a 4-way choice provides more information density than a binary choice.

### 2. Multi-Way Pairwise Decomposition
-   Ranking 4 talks is mathematically decomposed into 6 simultaneous pairwise outcomes (1st vs 2nd, 1st vs 3rd, ...).
-   This maximizes the utility of every screen you judge.

### 3. Active Learning Scheduler
The scheduler is the brain of the system, optimizing **next set selection** using a weighted sampling of five key factors:
-   **Exposure**: Prioritizes talks with the lowest total "rounds seen".
-   **Uncertainty**: Uses the "pairwise seen" count to mathematically favor talks with less statistical confidence.
-   **Novelty**: Penalizes pairs that have already appeared together, ensuring diverse matchups.
-   **Rating Closeness**: In "Exploit" mode (low Explore Rate), it prefers talks with similar ratings to refine the specific ordering.
-   **Track Mixing**: Mildly encourages cross-track comparisons to ensure the "AI" track is calibrated against the "Database" track, preventing isolated clusters.

## User-Defined Scoring

While the Elo engine works with raw rating numbers (e.g., 1400-1600), you can define your own preferred scale for the final output:

1.  **Set Scale**: In the main window control bar, set your **Min** (e.g. 1) and **Max** (e.g. 10) values.
2.  **View Scores**: When you click **Show Ranking**, a new **Score** column appears.
3.  **Automatic Normalization**: The app automatically maps the lowest current Elo rating to your Min and the highest to your Max, giving you an intuitive score distribution relative to the current field.

## Show Rank in Panels

You can stick to the "blind" evaluation (default) or toggle **Show Rank** to see the current standing of talks while comparing them:

-   **Toggle**: Use the checkbox in the control bar.
-   **Live Feedback**: Displays the current raw Elo and your custom Scaled Score below the abstract.
-   **Persistence**: Your preference is saved across sessions.

## Installation

1.  **Requirements**:
    -   Python 3.8+
    -   Tkinter (usually included with Python)
    -   Required packages (none, standard library only!)

2.  **Running the App**:
    ```bash
    python3 elo_talk_ranker.py --csv path/to/sessions.csv
    ```

## Usage Workflow

1.  **Launch**: Start the app with your CSV file containing columns `Talk ID`, `Title`, `Speaker`, `Abstract` (or similar headers).
2.  **Compare**: You will see a set of talks side-by-side.
3.  **Read & Judge**: Read the titles and abstracts.
4.  **Rank**: Assign a rank to each talk (1 = Best, 2 = Second Best, etc.).
    -   *No Ties Policy*: By default, each talk gets a unique rank (1 to N).
    -   *Ties*: Can be enabled/disabled via checkbox.
5.  **Submit**: Press `Shift+Enter` or click "Submit ranking". The Elo ratings update immediately.
6.  **Repeat**: Continue through sets until you are satisfied with the stability of the rankings.

### Controls

-   **Talks per comparison**: Switch between showing 2, 3, or 4 talks at a time.
-   **Exploration Rate**: (Slider)
    -   *Low (Left)*: Focus on "close calls" (talks with similar ratings) to refine the leaderboard order.
    -   *High (Right)*: Show random or unrated talks to discover new candidates.
-   **Target Appearances**: Set a goal for how many times you want to see each talk.

### Font Settings

Customize the reading experience via **View -> Font Settings...**.

![Font Settings](screenshots/placeholder_font_settings.png)

-   Change **Font Family** and **Size** for Titles, Speakers, and Abstracts independently.
-   Adjust **Abstract Spacing** and **Title Spacing** to improve readability on large screens.

### Shortcuts

-   **`1` ... `4`**: Assign rank to the corresponding panel.
    -   *Logic*: The first time you press a number (e.g. `1`), the talk in Panel 1 gets the best available rank. Pressing it again might change it? 
    -   *Correct Logic*: Pressing `1` assigns the *next available rank* to **Panel 1**. 
    -   *Example*: Quick ranking: Press `3` (Panel 3 is #1), then `1` (Panel 1 is #2), then `2` (Panel 2 is #3).
-   **`Shift + Enter`**: Submit the current ranking.
-   **`Esc`**: Clear all ranks.

## Exporting

Click **Export CSV** to control exactly what and how you export. A configuration dialog allows you to:

1.  **Select Columns**: Choose to include/exclude:
    -   *Rank* (1..N)
    -   *Elo Rating* (Raw sorting value)
    -   *User Score* (Your scaled 1-10 or custom score)
    -   *Metadata*: ID, Speaker, Title, Track
2.  **Sort Order**:
    -   *By Rank*: The standard leaderboard (Best talks first).
    -   *By ID*: Useful for exporting back to a spreadsheet where you need to match original IDs.

### Round Ranking
You can also enable **"Round Ranking"** in the main window (Display Options). When checked:
-   Elo ratings and User Scores are displayed as **whole numbers** (e.g. 1542, 9) in both the UI and Export.
-   Exported CSV values are also rounded to integers.
-   *Note*: Internal calculations always use full precision.