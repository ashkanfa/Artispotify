"""Functions used for interacting with data from the Spotify API."""


# Set up Spotipy (Spotify API package)
import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
#from spotify_credentials import *
client_id = os.environ["SPOTIFY_CLIENT_ID"]
client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
from collections import defaultdict



def df_listcell(input_list):
    """Helper function to store lists in individual dataframe cells."""
    # Create an empty series object and put the list in the first position
    listcell = pd.Series([], dtype='object')
    listcell[0] = input_list
    return listcell[0]



def chunks(input_list, n):
    """Yield successive n-sized chunks from a list to iterate past API endpoint limits."""
    for i in range(0, len(input_list), n):
        yield input_list[i:i+n]



def search_spotify(query):
    """Find artist name, id, and image from a search query."""
    # Set credentials and run the query
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    results = sp.search(q=query, type='artist')

    # Return the results as a tuple
    return [(x['name'], x['id'], x['images']) for x in results['artists']['items']]



def get_random_artists():
    """Generate a list of random artists."""
    # Set credentials
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    # Pull 2000 artsits randomly from spotify's catalog
    random_artists = []
    for i in range(0, 2000, 50):
        results = sp.search(q='year:0000-9999', limit=50, offset=i, type='artist')
        random_artists.extend([(x['name'], x['id'], x['followers']['total']) for x in results['artists']['items']])

    # Remove duplicates
    random_artists = list(set(random_artists))
    return random_artists



def playlist_df(playlist_id):
    """Given a playlist id, put relevant info into a dataframe."""
    # Set credentials and get playlist
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    pl = sp.playlist(playlist_id)

    # Pull out the relevant playlist information
    track_obj = pl['tracks']
    track_id = []
    for trk in track_obj['items']:
        track_id.append(trk['track']['id'])

    # Go through the pagination for the rest of the data
    while track_obj['next']:
        track_obj = sp.next(track_obj)
        for trk in track_obj['items']:
            track_id.append(trk['track']['id'])

    # Put results into a dataframe
    pl_df = pd.DataFrame({'Track_ID':track_id})
    pl_df['Track_Position'] = pl_df.index + 1
    return pl_df



def artist_df(artist_id_list):
    """Given a list of artist ids, put relevant info into a dataframe."""
    # Check if the input is not already a list
    if not isinstance(artist_id_list, list):
        artist_id_list = [artist_id_list]

    # Set credentials
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    art_df_list = []

    # Break the list into chunks of 50 and iterate over them
    chunked = list(chunks(artist_id_list, 50))
    for chunk in chunked:
        arts = sp.artists(chunk)

        # Pull out the relevant artist information
        for art in arts['artists']:
            art_dict = {'Artist_Name':art['name'],
                        'Artist_ID':art['id'],
                        'Artist_Genres':df_listcell(art['genres']),
                        'Artist_Followers':art['followers']['total'],
                        'Artist_Popularity':art['popularity']}
            art_df_list.append(art_dict)

    # Put results into a dataframe
    art_df = pd.DataFrame(art_df_list)
    return art_df



def album_df(album_id_list):
    """Given a list of album ids, put relevant info into a dataframe."""
    # Check if the input not already a list
    if not isinstance(album_id_list, list):
        album_id_list = [album_id_list]

    # Set credentials
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    alb_df_list = []

    # Break the list into chunks of 20 and iterate over them
    chunked = list(chunks(album_id_list, 20))
    for chunk in chunked:
        albs = sp.albums(chunk)

        # Pull out the relevant album information
        for alb in albs['albums']:
            alb_dict = {'Album_Name':alb['name'],
                        'Album_ID':alb['id'],
                        'Album_Type':alb['album_type'],
                        'Album_Artists':df_listcell([x['id'] for x in alb['artists']]),
                        'Album_Genres':df_listcell(alb['genres']),
                        'Album_Popularity':alb['popularity'],
                        'Album_Label':alb['label'],
                        'Album_Release_Date':alb['release_date']}
            alb_df_list.append(alb_dict)

    # Put results into a dataframe
    alb_df = pd.DataFrame(alb_df_list)
    return alb_df



def track_df(track_id_list):
    """Given a list of track ids, put relevant info into a dataframe (including audio features)."""
    # Check if the input is not already a list
    if not isinstance(track_id_list, list):
        track_id_list = [track_id_list]

    # Set credentials
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    trk_df_list = []
    trk_feat_df_list = []

    # Break the list into chunks of 50 and iterate over them
    chunked = list(chunks(track_id_list, 50))
    for chunk in chunked:
        trks = sp.tracks(chunk)
        trks_feat = sp.audio_features(chunk)

        # Pull out the relevant track information
        for trk in trks['tracks']:
            trk_dict = {'Track_Name':trk['name'],
                        'Track_ID':trk['id'],
                        'Track_Artists':df_listcell([x['id'] for x in trk['artists']]),
                        'Track_Album_Name':trk['album']['name'],
                        'Track_Album_ID':trk['album']['id'],
                        'Track_Popularity':trk['popularity'],
                        'Track_Explicitness':int(trk['explicit'] == True),
                        'Track_Duration':trk['duration_ms']}
            trk_df_list.append(trk_dict)

        # Pull out the relevant track feature information
        for trk_feat in trks_feat:
            if trk_feat is None:
                trk_feat_dict = {'Track_Key':None,
                                 'Track_Mode':None,
                                 'Track_TimeSig':None,
                                 'Track_Acousticness':None,
                                 'Track_Danceability':None,
                                 'Track_Energy':None,
                                 'Track_Instrumentalness':None,
                                 'Track_Liveness':None,
                                 'Track_Loudness':None,
                                 'Track_Speechiness':None,
                                 'Track_Valence':None,
                                 'Track_Tempo':None}
            else:
                trk_feat_dict = {'Track_Key':trk_feat['key'],
                                 'Track_Mode':trk_feat['mode'],
                                 'Track_TimeSig':trk_feat['time_signature'],
                                 'Track_Acousticness':trk_feat['acousticness'],
                                 'Track_Danceability':trk_feat['danceability'],
                                 'Track_Energy':trk_feat['energy'],
                                 'Track_Instrumentalness':trk_feat['instrumentalness'],
                                 'Track_Liveness':trk_feat['liveness'],
                                 'Track_Loudness':trk_feat['loudness'],
                                 'Track_Speechiness':trk_feat['speechiness'],
                                 'Track_Valence':trk_feat['valence'],
                                 'Track_Tempo':trk_feat['tempo']}
            trk_feat_df_list.append(trk_feat_dict)

    # Put results into a dataframe
    trk_df = pd.DataFrame(trk_df_list).join(pd.DataFrame(trk_feat_df_list))
    # Drop rows without audio feature data
    if 'Track_Key' in trk_df.columns:
        trk_df = trk_df[trk_df['Track_Key'].notna()].reset_index(drop=True)
    return trk_df



def artist_albumlist(artist_id):
    """Given an artist id, return the id of all albums as a list."""
    # Set credentials
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    # Loop through album types and get info from API
    album_types = ['album', 'single', 'compilation'] # exclude 'appears_on'
    results = []
    for typ in album_types:
        art_alb = sp.artist_albums(artist_id, album_type=typ, limit=50)

        # Put album name and id into a list and go through pagination
        results.extend([(x['name'], x['id']) for x in art_alb['items']])
        while art_alb['next']:
            art_alb = sp.next(art_alb)
            results.extend([(x['name'], x['id']) for x in art_alb['items']])
    return results



def album_tracklist(album_id):
    """Given an album id, return the id of all tracks as a list."""
    # Set credentials and get info from API
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    alb_trk = sp.album_tracks(album_id, limit=50)

    # Put results into a list and go through pagination
    results = [(x['name'], x['id']) for x in alb_trk['items']]
    while alb_trk['next']:
        alb_trk = sp.next(alb_trk)
        results.extend([(x['name'], x['id']) for x in alb_trk['items']])
    return results



def artist_tracklist(artist_id):
    """Given an artist id, return the id of all tracks as a list.

    Calling artist_albumlist() followed by album_tracklist() would unnecessarily
    reset the credentials over and over again, rather than just once here.
    """
    # Set credentials
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    # Loop through album types and get artist's albums from API
    album_types = ['album', 'single', 'compilation'] # exclude 'appears_on'
    albumlist = []
    for typ in album_types:
        art_alb = sp.artist_albums(artist_id, album_type=typ, limit=50)

        # Put data into a list and go through pagination
        albumlist.extend([x['id'] for x in art_alb['items']])
        while art_alb['next']:
            art_alb = sp.next(art_alb)
            albumlist.extend([x['id'] for x in art_alb['items']])

    # Pull the tracks for each album one-by-one
    tracklist = []
    for album in albumlist:
        # Get album tracks from API
        alb_trk = sp.album_tracks(album, limit=50)

        # Put results into a list and go through pagination
        tracklist.extend([(x['name'], x['id']) for x in alb_trk['items']])
        while alb_trk['next']:
            alb_trk = sp.next(alb_trk)
            tracklist.extend([(x['name'], x['id']) for x in alb_trk['items']])
    return tracklist



def related_artists_network(artist_id, degrees=0):
    """Given an artist id, return the id of all 20 related artists in a list, and their related artists in turn.

    artist_id - the Spotify ID of the seed artist
    degrees - the number of degrees out in the related artists network to search
    """
    # Set credentials
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    # Iterate over the artists for the number of degrees set (without retreading duplicates)
    unchecked = [artist_id] # id values which haven't been checked yet
    checked = [] # id values which have been checked
    artist_list = [] # passes intermediate results around and stores the final result
    while degrees > 0:
        for art in unchecked:
            related_artists = sp.artist_related_artists(art)
            related_ids = [x['id'] for x in related_artists['artists']]
            checked.append(art)
            artist_list.extend(related_ids)
        unchecked = list(set(artist_list).difference(checked))
        degrees -= 1
    artist_list = list(set(checked).union(unchecked))
    return artist_list



def recommended_tracks(artist_id_list, pop_list=range(5, 100, 30)):
    """Get a list of similar tracks based on a seed list o artists, distributed across popularity scores.

    artist_id_list - the list of artists with which to seed track recommendations
    pop_list - the target popularity scores for getting track recommendations
    """
    # Check if the input is not already a list
    if not isinstance(artist_id_list, list):
        artist_id_list = [artist_id_list]

    # Set credentials
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    # Generate a list of similar tracks for each artist, balanced by popularity score (if applicable)
    tracklist = []
    for art in artist_id_list:
        if pop_list:
            for pop in pop_list:
                recs = sp.recommendations(seed_artists=[art], limit=100, target_popularity=pop)
                tracklist.extend([x['id'] for x in recs['tracks']])
        else:
            recs = sp.recommendations(seed_artists=[art], limit=100)
            tracklist.extend([x['id'] for x in recs['tracks']])

    # Remove duplicate tracks and return the list
    tracklist = list(set(tracklist))
    return tracklist



def get_collabs(artist_id):
    """Get a list of collaborators for an artist id, based on who that artist has worked with in the past."""
    # Set credentials
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    # Loop through album types and get artist's albums from API
    album_types = ['album', 'single', 'compilation', 'appears_on']
    albumlist = []
    for typ in album_types:
        art_alb = sp.artist_albums(artist_id, album_type=typ, limit=50)

        # Put data into a list and go through pagination
        albumlist.extend([x['id'] for x in art_alb['items']])
        while art_alb['next']:
            art_alb = sp.next(art_alb)
            albumlist.extend([x['id'] for x in art_alb['items']])

    # Pull the tracks for each album one-by-one
    tracklist = []
    for album in albumlist:
        # Get album tracks from API
        alb_trk = sp.album_tracks(album, limit=50)

        # Put results into a list and go through pagination
        tracklist.extend([(x['name'], x['id'], x['artists']) for x in alb_trk['items']])
        while alb_trk['next']:
            alb_trk = sp.next(alb_trk)
            tracklist.extend([(x['name'], x['id'], x['artists']) for x in alb_trk['items']])

    # Extract the artist id's from the tracklist
    collab_list = []
    collab_list_filt = []
    for trk in tracklist:
        arts = trk[2]
        art_ids = [x['id'] for x in arts]
        collab_list.extend(art_ids)
        # Filter out songs not containing the original searched artist
        if artist_id in art_ids:
            collab_list_filt.extend(art_ids)

    # Drop the duplicates
    collab_list = list(set(collab_list))
    collab_list_filt = list(set(collab_list_filt))
    # Remove the original artist id and return both lists
    collab_list.remove(artist_id)
    collab_list_filt.remove(artist_id)
    return (collab_list, collab_list_filt)



def suggested_collabs(input_artist):
    """Return a list of suggested collaborations for a given seed artist."""
    # Get the related artist network (degree 1)
    net = related_artists_network(input_artist, 1)

    # Get the previous collaborators of the input artist
    seed_collabs = get_collabs(input_artist)
    seed_collabs = seed_collabs[1]

    # Get the previous collaborators of artists in the network
    # And keep track of which artist(s) the collaboration came from
    collabs = []
    worked_with = defaultdict(list)
    for art in net:
        # Get the name of the network artist
        art_df = artist_df(art)
        art_name = art_df['Artist_Name'][0]

        # Get the previous collaborators for this artist and add it to the overall list
        net_collabs = get_collabs(art)
        net_collabs = net_collabs[1]
        collabs.extend(net_collabs)

        # Add the network artist name to the list of artists associated with the suggestions
        for net_art in net_collabs:
            worked_with[net_art].append(art_name)

    # Remove duplicates and pre-existing collaborations
    collabs = list(set(collabs).difference(seed_collabs))

    # Get info about artists in the filtered list
    collabs_df = artist_df(collabs)

    # Add a column to keep track of which network artist(s) generated each recommendation
    new_col = []
    for art in range(collabs_df.shape[0]):
        source = worked_with[collabs_df.loc[art, 'Artist_ID']]
        # Remove duplicates and alphabetize
        source = list(set(source))
        new_col.append(source)
    collabs_df['Worked with'] = new_col

    # Sort the data by popularity and return the relevant information
    collabs_df = collabs_df.sort_values(by=['Artist_Popularity',
                                            'Artist_Followers'],
                                        ascending=False).reset_index(drop=True)
    collab_suggestions = pd.DataFrame({'Artist':collabs_df['Artist_Name'],
                                       'Popularity':collabs_df['Artist_Popularity'],
                                       'Follower Count':collabs_df['Artist_Followers'],
                                       'Worked with':collabs_df['Worked with']})
    return collab_suggestions
