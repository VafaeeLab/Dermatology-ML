# function to get the feature vectors of images
def get_features(all_images, all_labels, names):
    import time
    start = time.time()

    import numpy as np
    import pandas as pd
    import cv2
    from keras.applications.resnet import ResNet101
    from keras.preprocessing import image
    from keras.models import Model
    from keras.applications.resnet import preprocess_input

    df = pd.DataFrame({'Image':all_images, 'Labels': all_labels, 'Image_Names':names})
    base_model = ResNet101(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    image_size = 256
    #img_paths = img_df.Path.tolist()
    features_array = np.zeros((np.shape(df)[0],2048))

    for i, image_our in enumerate(df['Image']):
        img = image_our
        image_our = cv2.resize(image_our, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        #x = image.img_to_array(img)
        x = np.expand_dims(image_our, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        features = features.reshape(2048,)
        features_array[i,:] = features
    print('Running time: %.4f seconds' % (time.time()-start))
    df_features = pd.DataFrame(features_array)
    df_features['Image_name'] = df.Image_Names
    df_features['Labels'] = df.Labels
    return df_features

# svm
def model_svm():
    from sklearn.svm import SVC
    svm = SVC(gamma='scale', class_weight='balanced',probability = True)
    return svm 

# random forest
def model_rf():
    from sklearn.ensemble import RandomForestClassifier
    Random = RandomForestClassifier(n_estimators=120, criterion='gini', max_depth=None, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                               max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, 
                               random_state=None, verbose=0)
    return Random

# adaboost
def model_ada():
    from sklearn.ensemble import AdaBoostClassifier
    ADA = AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=0.1)
    return ADA

# KNN
def model_knn():
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    return knn