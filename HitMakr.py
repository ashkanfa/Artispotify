"""Codebase for the Streamlit front-end dashboard."""


# Import relevant libraries (many imports already contained within other scripts)
import streamlit as st
import altair as alt
from spotify_API import *
from ML_tool import *
from visual import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold



# Links to the Spotify API documentation
features_link = '[Audio Features](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/)'
track_link = '[Track Details](https://developer.spotify.com/documentation/web-api/reference/tracks/get-track/)'
# Text for the different parts of the dashboard
instructions_txt = '**Instructions:** HitMakr is a tool for helping artists on Spotify identify new ways to grow their following. \
                    To begin, search for your name in the sidebar and click on the button for the corresponding option. \
                    The program will then pull data from the Spotify API in order to identify which features of songs from similar \
                    artists drive popularity with the fanbase. Using this information, HitMakr will suggest songs in your \
                    library that are underperforming expectations and should be promoted more. It will also highlight the audio \
                    features that drive popularity up and down. Lastly, the program will provide a list of possible collaborations \
                    based on who similar artists have collaborated with in the past.'
promote_txt = 'Based on the model of which types of songs are most popular for artists similar to you, the following tracks in your \
               library are underperforming expectations. In promoting songs (e.g. on social media) to grow your following, these songs \
               should be given extra consideration. The table below is sorted by popularity score (which ranges from 0-100) and then \
               alphabetically by album. Click on the arrows to the top-right of the table to expand to full-screen. Columns can be \
               re-sorted by clicking on the headers.'
feature_txt = 'The chart and table below show the list of audio features identified as critical for driving popularity up or down for songs \
               made by similar artists. Positive values (blue bars) indicate features associated with *increased* popularity, while negative \
               values (orange bars) indicate features associated with *decreased* popularity. The small black lines indicate how variable the \
               importance of these features are in the model. Further explanation of these features is available at the Spotify API \
               documentation for {} and {}.'.format(features_link, track_link)
collab_txt = 'Below you will find a list of artists that have collaborated with other artists similar to you, along with the popularity \
              score and follower count for those artists. For convenience, the list of which similar artists they have collaborated with \
              is included as well. Click on the arrows to the top-right of the table to expand to full-screen. Columns can be re-sorted \
              by clicking on the headers.'
# Establish a default error message in case of network disconnects
error_txt = 'An error occurred. Please check your internet connection and try again.'

Popular_txt = 'Current top 50 popular songs similar to artist. The name of the artists are on the Y-axis with the corresponding popularity (0-100) on \
               the X-axis. Hover your mouse on the bars to disclose the name of the tracks.'

model_txt = 'From the Spotify API, given a seed artist, a network of related artist is pulled. Then Spotify \
            song recommendation runs on top of this network to get a pool of recommended tracks. These tracks act as training set for the machine \
            learning model. Each track in Spotify is identified by variables such as audio features and popularity. In this setting, the popularity metric is transformed to a binary \
            class by some decision rule. For building a predictive model, a random forest model is trained on popularity and audio \
            features. Once the model is trained, If we take the artist library as the test set, then False positive or tracks expected to be popular but currently \
            currently not, are translated into underperforming tracks with high potential for future promotions. The feature importance in the trained model is \
            used as audio trend in the artist market segment. The model was tuned for precision and on the preliminary analysis on random seeds of artists the precision was 0.72. \
            The performance of the model is dynamic since for every artist the training data will change.'


# Set up the main header text
st.title('HitMakr')
st.markdown('**by Ashkan Farahani**')
st.markdown('**ashkan.farahani@gmail.com**')
st.markdown(instructions_txt)
st.markdown('***Note:*** This program requires a stable internet connection and may take a few minutes to complete running.')
selected_header = st.subheader('*(No Artist Selected)*')



# Add search functionality to the sidebar
input_artist = None
search_box = st.sidebar.text_input('Search for artists')
if search_box:
    st.sidebar.header('Choose from below:')
    # Pull possible results to choose from
    search_results = search_spotify(search_box)
    for n, res in enumerate(search_results):
        # Display a selection button and image for each artist option
        if st.sidebar.button(res[0], key='search{}'.format(n)):
            selected_header.header('**Selected Artist:** *{}*'.format(res[0]))
            input_artist = res[1]
        if res[2]:
            st.sidebar.image(res[2][2]['url'], caption=res[0], width=160)
        else:
            st.sidebar.markdown('*No image found* :grey_exclamation:')
        st.sidebar.markdown('\n')
        st.sidebar.markdown('\n')



# Activate the rest of the dashboard when an artist is selected
if input_artist:
    # Use try-except to catch network disconnect errors
    try:
        # Set up the header text for this section and the loading message
        st.subheader('**Current Popular Songs:**')
        st.markdown(Popular_txt)
        loading_msg = st.warning('Loading & processing data. This may take a few minutes...')

        # Get the artist network and track data from the Spotify API for the input artist
        seed_results = seed_data(input_artist)

        # Pull the relevant information out of the seed data
        reclist_df = seed_results[5]

        # ['Track_Name', 'Track_ID', 'Track_Artists', 'Track_Album']
        # 'Track_Popularity'
        onTrend_df = reclist_df[['Track_Name', 'Track_Artists', 'Track_Popularity']]
        # onTrend_df.sort_values(by = ['Track_Popularity'], ascending = False, inplace = True)
        # onTrend_df = onTrend_df.iloc[:20]
        onTrend_df_nl = onTrend_df.nlargest(50, 'Track_Popularity')
        onTrend_df_nl['Artist'] = onTrend_df_nl.apply(lambda x: get_name(x['Track_Artists'][0]), axis =1)
        onTrend_df_nl.drop('Track_Artists', axis=1, inplace= True)
        #onTrend_df_nl = onTrend_df_nl.set_index(conTrend_df_nl.index + 1)
        #onTrend_df_nl.drop('Track_Artists', axis=1)


        bars_oT = alt.Chart(onTrend_df_nl, width=700, height=900).mark_bar().encode(
        x=alt.X("Track_Popularity:Q", title="Track Popularity"),
        y=alt.Y("Artist:N", title="Artist", sort = '-x'),
        tooltip = ['Track_Name']
        )


        # text = alt.Chart(onTrend_df_nl, ).mark_text(align = 'left', dx=-225, color='white').encode(
        # x=alt.X("Track_Popularity:Q", title="Track Popularity"),
        # y=alt.Y("Artist:N", title="Artist", sort = '-x'),
        # text=alt.Text('Track_Name:N')
        # )

        # text = bars_oT.mark_text(
        # align='left',
        # baseline='middle',
        # dx=3  # Nudges text to right so it doesn't appear on top of the bar
        # ).encode(
        # text='Track_Name:N'
        # )

        #(bars_oT + text).properties(height=900)

        st.write(bars_oT)

        #onTrend_df_nl = onTrend_df_nl[['Artist', 'Track_Name', 'Track_Popularity']]
        #st.dataframe(onTrend_df_nl)

        #######################################################################################################################
        st.subheader('**Songs to Promote:**')
        st.markdown(promote_txt)
        artist_library_df = track_df(seed_results[2])
        # Update the error message if the artist library is too small to have related artists yet,
        # which will automatically exit the program due to an error during prep_data_streamlit()
        if reclist_df is None:
            error_txt = '**Error:** Artist library too small, no related artists found.'
        # Generate the training and test data
        X_train, y_train, X_test, y_test = prep_data_streamlit(artist_library_df, reclist_df)

        # Set up and fit the model
        RFC = RandomForestClassifier(class_weight='balanced_subsample', n_estimators=100, max_depth=2, random_state=0)


        #########################################################################################################
        # define models and parameters
        #RFC = RandomForestClassifier(class_weight='balanced_subsample')

        # summarize results
        #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        clf, cols2scale, cols2drop = build_pipeline(RFC)
        #clf.fit(X_train, y_train)
        #n_estimators = [100, 500, 1000]
        #max_features = ['sqrt', 'log2']
        # define grid search
        #grid = dict(model__n_estimators=n_estimators,model__max_features=max_features)
        #cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        #grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=cv, scoring='precision',error_score=0)
        #grid_result = grid_search.fit(X_train, y_train)
        #precision = grid_result.best_score_
        # Tuned model
        #tuned_RFC = RandomForestClassifier(class_weight='balanced_subsample', n_estimators = grid_result.best_params_['model__n_estimators'], max_features = grid_result.best_params_['model__max_features']  )
        #grid_result.best_params_['model__n_estimators']
        #grid_result.best_params_['model__max_features']


        clf.fit(X_train, y_train)
        # Generate suggested songs to promote based on y_pred
        y_pred = clf.predict(X_test)
        song_suggestions = songs_to_promote(artist_library_df, y_test, y_pred)


        # Re-index the dataframe so it starts at 1 for better readability
        song_suggestions = song_suggestions.set_index(song_suggestions.index + 1)
        # Display the results
        loading_msg.text('')
        st.dataframe(song_suggestions)
        #st.markdown('Precision:{}'.format(precision))
        #st.subheader('**The Most Popular Tracks:**')
        # popular_tracks = reclist_df.loc[['Track_Name', 'Track_Artists', 'Track_Popularity']]
        # popular_tracks = popular_tracks.sort_values(by=['Track_Popularity',
        #                                               'Track_Artists',
        #                                               'Track_Name'],
        #                                           ascending=[False, True, True])
        # song_suggestions = pd.DataFrame({'Popularity':suggestion_df['Track_Popularity'],
        #                                  'Album':suggestion_df['Track_Album_Name'],
        #                                   'Track':suggestion_df['Track_Name']}).reset_index(drop=True)

        # popular_tracks_20 = popular_tracks.iloc[:20]
        #loading_msg.text('')
        #st.dataframe(reclist_df.iloc[:3])



        # Set up the header text for this section
        st.subheader('**On-Trend: Audio Features Driving Popularity:**')
        st.markdown(feature_txt)

        # Pull out the model components
        col_trans = clf['preprocess']
        forest = clf['model']
        # Generate the feature names
        new_cols = [c for c in X_train.columns if c not in cols2scale+cols2drop]
        new_cols = cols2scale + new_cols
        col_labels = [x.replace('Track_', '') for x in new_cols]

        # Preprocess X_train for calculating feature importance
        X_trans = col_trans.fit_transform(X_train)
        X_trans = pd.DataFrame({k:X_trans[:,n] for n, k in enumerate(col_labels)})
        # Calculate and plot the Random Forest feature importances
        sorted_mean, sorted_std, sorted_labels, sorted_colors = get_RFC_importances(forest, X_trans, y_train, col_labels)
        importances, fig = plot_RFC_importances(sorted_mean, sorted_std, sorted_labels, sorted_colors,st_xlabels=True)

        altair_df = pd.DataFrame({'importance':sorted_mean,
                      'feature':sorted_labels})

        #altair_df.sort_values(['importance'],ascending = False, inplace = True)

        bars = alt.Chart(altair_df,width=700, height=600).mark_bar().encode(
            x=alt.X("feature:N", title="Audio Features",sort=None),
            y=alt.Y("importance:Q", title="Popularity"),
            color=alt.condition(
                alt.datum.importance > 0,
                alt.value("steelblue"),
                alt.value("orange")
            ),
            tooltip = ['feature']
        )
        st.write(bars)
        #################################################################################################
                # Create a dataframe of the results
        # mean_disp_list = []
        # msg_list = []
        # for imp in sorted_mean:
        #     if imp < 0:
        #         mean_disp = '- {:4.1f}%'.format(abs(imp)*100)
        #         msg = 'Drives it DOWN'
        #     else:
        #         mean_disp = '+ {:4.1f}%'.format(imp*100)
        #         msg = 'Drives it UP'
        #     mean_disp_list.append(mean_disp)
        #     msg_list.append(msg)
        # importances = pd.DataFrame({'Feature':sorted_labels, 'Impact on Popularity':msg_list, 'Importance':mean_disp_list})
        #
        # # Plot the data and format the plot
        # plt.style.use('fivethirtyeight')
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.bar(range(len(sorted_labels)), sorted_mean, yerr=sorted_std, color=sorted_colors)
        # ax.set_title('Important Audio Features', fontsize=18, fontweight="bold")
        # # if st_xlabels:
        # #ax.xticks(range(len(sorted_labels)), range(1, len(sorted_labels) + 1), fontsize=14)
        # ax.set_xlabel('Feature # (see below)', fontsize=14)
        # # else:
        # #     ax.xticks(range(len(sorted_labels)), sorted_labels, rotation=75, fontsize=16)
        # ax.set_ylabel('Relative Importance', fontsize=14)
        #ax.yticks(fontsize=14)
        #plt.show()

        ############################################################################
        # Re-index the dataframe so it starts at 1 for better readability
        importances = importances.set_index(importances.index + 1)
        # Display the results
        #st.pyplot(fig)
        st.table(importances)



        # Set up the header text for this section and the loading message
        st.subheader('**Possible Collaborations:**')
        st.markdown(collab_txt)
        loading_msg = st.warning('Loading & processing data. This may take a few minutes...')

        # Get the suggested collaborations for the input artist
        collab_suggestions = suggested_collabs(input_artist)

        # Re-index the dataframe so it starts at 1 for better readability
        collab_suggestions = collab_suggestions.set_index(collab_suggestions.index + 1)
        # Display the results
        loading_msg.text('')
        st.dataframe(collab_suggestions)

        # How the model works
        st.subheader('**Machine Learning Approach:**')
        st.markdown(model_txt)



    except:
        # Return an error message if something goes wrong
        loading_msg.text('')
        st.error(error_txt)
