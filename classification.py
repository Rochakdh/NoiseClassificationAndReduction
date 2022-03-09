def classify(FILE_NAME):

    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import librosa
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder
    import paths_src

    new_df = pd.read_csv(paths_src.READ_CSV_AUDIO_EXTRACTOR)
    new_df.head()

    y=np.array(new_df['class'].tolist())


    labelencoder=LabelEncoder()
    y=to_categorical(labelencoder.fit_transform(y))


    new_model = keras.models.load_model(paths_src.CLASSIFICAATION_MODEL)

    #provide input audio file to classify here
    #for recorded
    if(FILE_NAME == paths_src.RECORDED_FOLDER_PATH):
        filename = f"{FILE_NAME}/{paths_src.RECORDED_FILE_NAME}"
    #for browse
    else:
        filename = f"{paths_src.BROWSE_PATH}/{FILE_NAME.name}"

    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    print(mfccs_scaled_features)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    print(mfccs_scaled_features)
    print(mfccs_scaled_features.shape)
    predicted_label= np.argmax(new_model.predict(mfccs_scaled_features),axis=1)


    print(predicted_label)
    prediction_class = labelencoder.inverse_transform(predicted_label) 
    print(prediction_class)

    return prediction_class