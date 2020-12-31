"""Functions used for modeling data."""


# Import libraries and functions
import os
import pickle
import numpy as np
import pandas as pd
from spotify_API import *

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics



def pop_classes(pop_vals, cutoffs=[75]):
    """Turn popularity scores into classes based on percentile cutoffs.

    pop_vals - the list of popularity values
    cutoffs - the list of percentile cutoffs to use
    (will assign a separate class for each percentile in order)
    """
    # Check if the cutoff is not already a list
    if not isinstance(cutoffs, list):
        cutoffs = [cutoffs]

    # Convert popularity values to a numpy array (for speed)
    p = np.array(pop_vals)

    # Set each percentile chunk to a different class (in ranked order)
    classes = np.array([0] * len(p))
    for pct in cutoffs:
        c = np.where(p >= np.percentile(p, pct), 1, 0)
        classes = classes + c
    return classes



def seed_data(artist_id, degrees=2):
    """Pull the relevant data for a seed artist.

    artist_id - the Spotify ID of the seed artist
    degrees - the number of degrees out in the related artists network to search
    """
    # Get the network of related artists
    net = related_artists_network(artist_id, degrees)

    # Get the seed artist's tracklist id values
    seed_list = artist_tracklist(artist_id)
    seed_list = [x[1] for x in seed_list]

    # Get the list of recommended tracks and remove any belonging to the seed artist
    recs = recommended_tracks(net)
    recs_filt = list(set(recs).difference(seed_list))

    # Get the full dataframe for each track id in the recommendation list
    if recs_filt:
        df = track_df(recs_filt)
    else:
        # Assign None in case the artist is so small there weren't any recommended artists
        df = None
    return (artist_id, net, seed_list, recs, recs_filt, df)



def save_random_artist_data(start_idx=0, end_idx=3):
    """Go through a slice of the random_artists seed list and generate/save the data needed for modeling tests."""
    # Load the random_artists list, or create & save it if it doesn't exist
    if os.path.exists('Data/random_artists.pkl'):
        print('Load: random_artists')
        with open('Data/random_artists.pkl', 'rb') as f:
            random_artists = pickle.load(f)
    else:
        random_artists = get_random_artists()
        with open('Data/random_artists.pkl', 'wb') as f:
            pickle.dump(random_artists, f)
        print('Saved: random_artists')

    # Get the seed_data for each artist and save it
    for n, artist in enumerate(random_artists[start_idx:end_idx]):
        save_name = 'data_artist_{}.pkl'.format(n+start_idx)
        save_path = 'Data/{}'.format(save_name)

        # Skip this file if it already exists
        if os.path.exists(save_path):
            print('Skipped: ', save_name)
            continue

        # Create and save the data
        save_data = seed_data(artist[1])
        with open(save_path, 'wb') as f:
            pickle.dump([save_name, artist, save_data], f)
        print('Saved: ', save_name)



def load_sample_data(filerange=range(201)):
    """Load the saved example artist data for evaluation."""
    # Load the data
    results_list = []
    for i_file in filerange:
        fpath = 'Data/data_artist_{}.pkl'.format(i_file)
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'rb') as f:
            results = pickle.load(f)
        results_list.append(results)
    return results_list



def drop_cols(input_df):
    """Drop the irrelevant columns of the input dataframe."""
    # Drop the columns
    if 'Track_Album' in input_df.columns:
        all_features = input_df.drop(['Track_Name', 'Track_ID', 'Track_Artists', 'Track_Album'], axis=1)
    else:
        all_features = input_df.drop(['Track_Name', 'Track_ID', 'Track_Artists', 'Track_Album_Name', 'Track_Album_ID'], axis=1)
    return all_features



def split_df(input_df):
    """Generate the training and test splits from a dataframe, plus shuffled data for baseline."""
    # Convert columns to relevant X and y features
    all_features = drop_cols(input_df)
    X = all_features.drop(['Track_Popularity'], axis=1)
    y_vals = all_features['Track_Popularity']

    # Convert popularity values to binary classes
    y = pop_classes(y_vals)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.15)

    # Create a randomly shuffled version of 'y' to act as a baseline comparison
    y_train_shuffled = np.array(y_train)
    y_test_shuffled = np.array(y_test)
    np.random.shuffle(y_train_shuffled)
    np.random.shuffle(y_test_shuffled)

    return (X_train, X_test, y_train, y_test, y_train_shuffled, y_test_shuffled)



def make_RFC_list(n_est_list, max_depth_list):
    """Create a list of different Random Forest Classifier models to search through.

    Note: GridSearchCV wasn't used because it doesn't return all estimators for comparing
    hold-out scores, only the best performing one.
    """
    # Check if n_est_list is not already a list
    if not isinstance(n_est_list, list):
        n_est_list = [n_est_list]
    # Check if max_depth_list is not already a list
    if not isinstance(max_depth_list, list):
        max_depth_list = [max_depth_list]

    models = []
    for est in n_est_list:
        for depth in max_depth_list:
            models.append(RandomForestClassifier(class_weight='balanced_subsample',
                                                 n_estimators=est,
                                                 max_depth=depth,
                                                 random_state=0))
    return models



def make_LR_list(penalty_list, c_list):
    """Create a list of different Logistic Regression models to search through.

    Note: GridSearchCV wasn't used because it doesn't return all estimators for comparing
    hold-out scores, only the best performing one.
    """
    # Check if penalty_list is not already a list
    if not isinstance(penalty_list, list):
        penalty_list = [penalty_list]
    # Check if c_list is not already a list
    if not isinstance(c_list, list):
        c_list = [c_list]

    models = []
    for pen in penalty_list:
        for c_val in c_list:
            models.append(LogisticRegression(class_weight='balanced',
                                             penalty=pen,
                                             C=c_val,
                                             solver='saga',
                                             random_state=0))
    return models



def make_SVC_list(c_list):
    """Create a list of different Support Vector Classifier models to search through.

    Note: GridSearchCV wasn't used because it doesn't return all estimators for comparing
    hold-out scores, only the best performing one.
    """
    # Check if c_list is not already a list
    if not isinstance(c_list, list):
        c_list = [c_list]

    models = []
    for c_val in c_list:
        models.append(SVC(kernel='linear',
                          class_weight='balanced',
                          C=c_val,
                          random_state=0))
    return models



def build_pipeline(input_model):
    """Build the pipeline object for classification."""
    # Select the columns to be re-scaled and dropped
    cols2scale = ['Track_Duration', 'Track_Loudness', 'Track_Tempo']
    cols2drop = ['Track_Key', 'Track_TimeSig']

    # Set up the appropriate column transformer for preprocessing
    ct = ColumnTransformer([('scaler', MinMaxScaler(), cols2scale),
                            ('drop_cols', 'drop', cols2drop)],
                           remainder='passthrough')

    # Set up the pipeline object
    pipeline = Pipeline([('preprocess', ct),
                         ('model', input_model)])
    return pipeline, cols2scale, cols2drop



def run_cv(input_model, X_train, y_train):
    """Run the cross-validation on the input model."""
    # Set up the pipeline object
    pipeline, cols2scale, cols2drop = build_pipeline(input_model)

    # Run the cross-validation and return the results
    cv_results = cross_validate(pipeline,
                                X_train,
                                y_train,
                                scoring='recall',
                                cv=5,
                                return_train_score=True,
                                return_estimator=True)
    return cv_results



def prep_data_streamlit(artist_library_df, reclist_df):
    """Prepare the training and test data for use with the front-end.

    artist_library_df - the tracklist of the original seed artist with all metadata
    reclist_df - the tracklist of the recommended tracks with all the metadata
    """
    # Generate X_train and y_train based on the recommended tracks dataframe
    feats_train = drop_cols(reclist_df)
    X_train = feats_train.drop(['Track_Popularity'], axis=1)
    y_vals_train = feats_train['Track_Popularity']
    # Convert popularity values to binary classes
    y_train = pop_classes(y_vals_train)

    # Generate X_test and y_test based on the artist libarary dataframe
    feats_test = drop_cols(artist_library_df)
    X_test = feats_test.drop(['Track_Popularity'], axis=1)
    y_vals_test = feats_test['Track_Popularity']
    # Convert popularity values to binary classes
    y_test = pop_classes(y_vals_test)

    # Return the training and test data, and the artist library
    return X_train, y_train, X_test, y_test



def get_RFC_importances(forest, X_trans, y_train, col_labels):
    """Calculate and sort feaure importances from the random forest classifier.

    forest - the fitted random forest model
    X_trans - X_train used to fit the model, with preprocessing already applied
    y_train - y_train used to fit the model
    col_labels - names of the features, with preprocessing (e.g. dropping and resorting) already applied
    """
    # Calculate the mean and standard deviation of the feature importances
    imp_mean = forest.feature_importances_
    imp_std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    # Filter X_trans into positive and negative class samples, and take the mean across features
    pop1 = X_trans[y_train == 1]
    pop0 = X_trans[y_train == 0]
    mean1 = pop1.mean(axis=0)
    mean0 = pop0.mean(axis=0)

    # For classes where smaller values drive popularity, flip the sign and the bar color for that feature
    barcolors = ['b'] * len(imp_mean)
    for n in range(len(imp_mean)):
        if mean1[n] < mean0[n]:
            imp_mean[n] = -imp_mean[n]
            barcolors[n] = 'r'

    # Re-sort the relevant vectors by magnitude of feature importance
    ordered_idx = np.argsort(abs(imp_mean))[::-1]
    sorted_mean = imp_mean[ordered_idx]
    sorted_std = imp_std[ordered_idx]
    sorted_labels = col_labels.copy()
    sorted_colors = barcolors.copy()
    for i, n in enumerate(ordered_idx):
        sorted_labels[i] = col_labels[n]
        sorted_colors[i] = barcolors[n]

    # Return the mean, std, labels, and bar colors
    return sorted_mean, sorted_std, sorted_labels, sorted_colors



def songs_to_promote(artist_library_df, y_test, y_pred):
    """Use the results from the model to find false negatives,
    songs that should be popular but aren't, to suggest for promotion.

    artist_library_df - the tracklist of the original seed artist with all metadata
    y_test - the calculated popularity scores for the artist's library
    y_pred - the predicted popularity scores for the artist's library
    """
    # Filter out the predicted false negatives
    suggestion_df = artist_library_df[(y_test == 0) & (y_pred == 1)]

    # Sort the suggestions and return the relevant columns
    suggestion_df = suggestion_df.sort_values(by=['Track_Popularity',
                                                  'Track_Album_Name',
                                                  'Track_Name'],
                                              ascending=[False, True, True])
    song_suggestions = pd.DataFrame({'Popularity':suggestion_df['Track_Popularity'],
                                     'Album':suggestion_df['Track_Album_Name'],
                                      'Track':suggestion_df['Track_Name']}).reset_index(drop=True)
    return song_suggestions
