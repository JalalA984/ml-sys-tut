import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)

    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
def preprocess(ds):
    return ds / 255.0

x_train = preprocess(x_train)
x_test = preprocess(x_test)


# hyberband: uses adaptive resource allocation and early stopping
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

# reduce overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

# start search for hyperparams
tuner.search(x_train, y_train,
             epochs=30, validation_split=0.2,
             callbacks=[early_stop])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
model.fit(x_train, y_train, epochs=50, validation_split=0.2)


model.evaluate(x_test, y_test)