# Fun Datasets for Pandas/Python Practice (Ditch Titanic & Air Quality)

**Why this note exists:** You're in your Phase 2 catch-up stretch — Pandas / data-handling — right after clearing your NumPy exit gate. Titanic and Air Quality Index datasets are the "default" ones every tutorial uses because they're clean and well-documented, but they're boring and don't build any real intuition beyond `groupby()` and `fillna()`. Below is a big bundle of datasets that are just as good (often better) for learning real Pandas/EDA skills, but way more fun to poke around in. Each one lists: what it is, why it's actually useful for skill-building (not just "cool"), where to find it, and 2-3 concrete exercises you can run today.

---

## How to use this note

Don't just load a dataset and `.head()` it and move on. For every dataset, force yourself to do at least these 4 things — this is what actually builds Pandas fluency:

1. **Clean it** — find and fix missing values, duplicate rows, inconsistent types/casing
2. **Reshape it** — practice `groupby`, `pivot_table`, `merge`, `melt`
3. **Ask 3 questions of the data** — then answer them with code, not eyeballing
4. **Visualize 1 finding** — Matplotlib/Seaborn, one clean chart, properly labeled

Treat each dataset like a 45-90 minute mini-project, not a "look at it" exercise.

---

## Tier 1 — Best for a Pandas beginner (start here)

### 1. Pokémon Dataset
- **What:** Stats (HP, Attack, Defense, Speed, Type1/Type2, Generation, Legendary flag) for all Pokémon.
- **Where:** Kaggle — search "Pokemon with stats" (the Rounak Banik version is the most common one used in tutorials).
- **Why it's great for learning:** Two categorical columns (Type1/Type2) with nulls in Type2 force you to handle missing categorical data properly (not just numeric `fillna`). Great for `groupby` + aggregation practice.
- **Exercises:**
  - Which Type1 has the highest average Attack? Which has the highest average Defense-to-Speed ratio?
  - Build a "best overall stats" ranking using a weighted sum you define yourself — this forces you to create a new engineered column.
  - Are Legendary Pokémon *actually* statistically stronger, or is that a myth? Prove it with a groupby comparison, not vibes.

### 2. Spotify Tracks / Top Songs Dataset
- **What:** Audio features (danceability, energy, tempo, valence, loudness) + streams/popularity for thousands of tracks.
- **Where:** Kaggle — "Spotify Tracks Dataset" or "Top Spotify Songs" (several versions exist, any recent one works).
- **Why it's great:** Numeric feature-heavy — perfect for correlation analysis, scatter plots, and your first taste of "which features actually predict popularity" (a preview of ML feature importance, which you'll hit later in your roadmap).
- **Exercises:**
  - Correlation matrix: does higher "energy" or "danceability" correlate with popularity? Visualize with a heatmap.
  - Group by genre/artist and find who has the most consistent (lowest variance) song valence.
  - Bucket songs into tempo ranges (slow/medium/fast) using `pd.cut()` and compare average popularity per bucket.

### 3. Video Game Sales Dataset
- **What:** Global/regional sales figures for thousands of video games, by platform, genre, publisher, year.
- **Where:** Kaggle — "Video Game Sales."
- **Why it's great:** Real messy-ish data (some missing years/publishers), multiple region columns that are perfect for `melt()` practice (wide-to-long reshaping), which most tutorials skip.
- **Exercises:**
  - Melt the NA/EU/JP/Other sales columns into a single "region" + "sales" pair of columns, then re-aggregate.
  - Which genre dominated each decade? (groupby year-bucket + genre)
  - Pivot table: platform vs genre, values = total global sales.

---

## Tier 2 — Slightly richer, good for intermediate practice

### 4. Marvel / DC Superheroes Dataset
- **What:** Power stats, alignment (hero/villain), publisher, appearances count for thousands of characters.
- **Where:** Kaggle — "Superhero Database" or FiveThirtyEight's "comic-characters" dataset (also on GitHub).
- **Why it's great:** Has real inconsistent/dirty text data (character names, alignment spelled differently, missing power stats) — great for string cleaning with `.str` methods.
- **Exercises:**
  - Clean and standardize the alignment column (Good/Bad/Neutral have inconsistent casing/spellings in some versions).
  - Compare Marvel vs DC on average "appearances" — who has more one-hit-wonder characters vs recurring ones?
  - Find characters with the single highest "power gap" between their strongest and weakest stat.

### 5. World Happiness Report
- **What:** Country-level happiness score plus GDP per capita, social support, life expectancy, freedom, corruption perception — for ~150 countries, multiple years.
- **Where:** Kaggle — "World Happiness Report" (or directly from worldhappiness.report).
- **Why it's great:** Multi-year data means you can practice merging/joining across years and tracking change over time — a very real-world skill (time-series-lite without full time-series complexity).
- **Exercises:**
  - Merge multiple years into one DataFrame with a "year" column, then plot India's happiness score trend over time.
  - Which single factor correlates most strongly with happiness score? Is it really GDP, or something else?
  - Rank countries by "happiness relative to GDP" (i.e., punching above their economic weight) — this requires you to engineer a new ratio column.

### 6. Anime Dataset (MyAnimeList)
- **What:** Anime titles with genre, episodes, rating, member count, type (TV/Movie/OVA), studio.
- **Where:** Kaggle — "Anime Recommendations Database" or "MyAnimeList Dataset."
- **Why it's great:** Genre column often contains comma-separated multi-values in one cell (e.g. "Action, Adventure, Comedy") — this is a classic real-world messy pattern you'll hit constantly in actual jobs, and it forces you to learn `.str.split()` + explode.
- **Exercises:**
  - Explode the genre column so each anime-genre pair is its own row, then find the most common genre overall.
  - Do longer anime (more episodes) tend to have higher or lower average ratings?
  - Which studio has the most consistently high-rated catalog (highest average rating with at least 10 titles)?

### 7. F1 Racing / Formula 1 World Championship Dataset
- **What:** Every F1 race, driver, constructor, lap time, and result since 1950, split across multiple related CSVs (drivers, races, results, constructors, lap_times, etc.)
- **Where:** Kaggle — "Formula 1 World Championship (1950-2024)."
- **Why it's great:** This is a *relational* dataset (multiple linked tables), which is genuinely closer to real-world data work than any single-CSV dataset. Forces you to actually learn `merge()` properly with real foreign-key-style joins.
- **Exercises:**
  - Join drivers + results + races to build one flat table: driver name, race name, year, finishing position.
  - Find the driver with the best average finishing position per decade.
  - Compare constructors: who improved the most between two 5-year windows?

---

## Tier 3 — For when you want a real challenge (messier, bigger)

### 8. Steam Games Dataset
- **What:** Every game on Steam — price, tags, reviews, playtime, release date, supported platforms.
- **Where:** Kaggle — "Steam Games Dataset" or "Steam Store Games."
- **Why it's great:** Genuinely messy — nested tag lists, inconsistent price formatting, missing review data. Good stepping stone toward real unstructured cleaning work.
- **Exercises:**
  - Clean the price column (currency symbols, "Free to Play" text mixed with numbers) into a proper float column.
  - Which tag combination shows up most in the top 100 highest-reviewed games?
  - Plot number of games released per year — when did the "indie explosion" actually happen?

### 9. Chess Games Dataset (Lichess)
- **What:** Millions of real online chess games with moves, opening name, player ratings, result, time control.
- **Where:** Kaggle — "Chess Game Dataset (Lichess)."
- **Why it's great:** Large enough to force you to think about performance (chunking, efficient filtering) instead of just correctness. Good bridge toward "data at scale" instincts.
- **Exercises:**
  - Which openings have the highest win rate for white vs black?
  - Does a bigger rating gap between players predict a faster/shorter game?
  - Bucket games by time control (bullet/blitz/rapid/classical) and compare average game length.

### 10. Wine Quality / Whisky / Cocktail Datasets
- **What:** Chemical properties (acidity, sugar, alcohol %, pH) of wines mapped to a quality score — or, for the more fun version, a cocktail recipes dataset with ingredients and categories.
- **Where:** UCI ML Repository ("Wine Quality") for the serious version; Kaggle "Cocktails" dataset for the fun ingredient-based one.
- **Why it's great:** The wine dataset is a genuinely famous "regression practice" dataset in ML (you'll likely see it again once you hit Phase 3 regression work) — so this doubles as an early preview. The cocktails one is great for ingredient-list explode/groupby practice, similar to the anime genre trick.
- **Exercises (cocktails):**
  - Explode ingredient lists, find the single most common ingredient across all cocktails.
  - Which category (shot, cocktail, punch) has the most ingredient variety on average?

---

## Bonus: Where to actually get all of this

- **Kaggle Datasets** (kaggle.com/datasets) — by far the best source; searchable, most have a clean CSV, many have companion notebooks you can read *after* you've tried it yourself (don't peek first).
- **UCI Machine Learning Repository** — more "serious"/academic datasets, good once you want to blend fun practice with actual ML-prep datasets.
- **FiveThirtyEight's GitHub data repo** — journalistic datasets (sports, politics, pop culture) that are pre-cleaned and often come with the original article, so you can compare your findings to theirs.
- **Kaggle API** — once you're comfortable, learn to pull datasets straight via `kaggle datasets download -d <dataset-slug>` instead of manually downloading zips. Small skill, but it's the actual industry habit.

---

## A note on how to sequence this against your roadmap

Since you're in the Pandas/data-handling catch-up stretch of Phase 2, here's the order I'd actually run these in, matched to skill difficulty:

1. **Pokémon** → basic cleaning + groupby (day 1)
2. **Video Game Sales** → melt/reshape + pivot_table (day 1-2)
3. **Spotify** → correlation, numeric EDA, first visualization habits (day 2)
4. **Anime** → explode/multi-value columns, a genuinely common real-world pattern (day 3)
5. **F1** → multi-table merges — this is the one that will feel most "job-like" (day 3-4)
6. **Steam or Chess** → pick whichever sounds more fun, treat it as your Phase 2 exit-gate style project — messy, real, needs actual judgment calls

Don't do all of them back-to-back mechanically — pick 3-4 max, but do each one *properly* (all 4 steps from the top of this note). Depth beats breadth here; one dataset done with real rigor (cleaning + reshaping + questions + visualization) teaches you more than ten datasets skimmed.
