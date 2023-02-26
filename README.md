# movie-revenue-model
An ML powered model used to predict movies revenue and classify their success

## Preprocessing

- Handling multiple datasets:
  - Used the movie name as the primary key between the three tables to merge them into one dataframe.
  - Transformed the voice-actors database into one column to indicate whether its an animated movie or not.

- Director:
  - Used [`TMDB API`](https://developers.themoviedb.org/3/getting-started/introduction) to **fill missing directors**.
  - Replaced the director with his or her TMDB popularity score.

- Release Date:
  - Extracted day, month, season columns from it.
  - Transformed it to a scalar by converting the days to a fraction of a year.

- Movie revenue:
  - Used [`cpi`](https://palewi.re/docs/cpi/) library to adjust the movie revenue to inflation using the release date. 


## Libraries Used

- pandas
- numpy
- cpi
- skcikit-learn
- matplotlib
- seaborn
- pickle
