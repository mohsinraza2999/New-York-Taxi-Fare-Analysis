def add_features(df):
    df['DOLocationID'] = df['DOLocationID'].astype('str')
    df['PULocationID'] = df['PULocationID'].astype('str')
    df['pickup_dropoff'] = df['DOLocationID'] + ' ' + df['PULocationID']

    # Mean distance by pickup_dropoff
    dist_group = df[["pickup_dropoff", "trip_distance"]].groupby('pickup_dropoff').mean()
    dist_map = dist_group.to_dict()['trip_distance']
    df['mean_distance'] = df['pickup_dropoff'].map(dist_map)

    # Mean duration by pickup_dropoff
    dur_group = df[["pickup_dropoff", "duration"]].groupby('pickup_dropoff').mean()
    dur_map = dur_group.to_dict()['duration']
    df['mean_duration'] = df['pickup_dropoff'].map(dur_map)

    df['day'] = df['tpep_pickup_datetime'].dt.day_name()
    df['month'] = df['tpep_pickup_datetime'].dt.month_name()
    df['rush_hour'] = df['day'].apply(lambda x: 0 if x in ['Saturday', 'Sunday'] else 1)

    def rush_hourizer(row):
        hour = row['tpep_pickup_datetime'].hour
        return 1 if (6 <= hour <= 10) and row['day'] not in ['Saturday', 'Sunday'] else 0

    df['rush_hour'] = df.apply(rush_hourizer, axis=1)

    return df