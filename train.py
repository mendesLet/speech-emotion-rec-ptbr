from tensorflow.keras.utils import to_categorical
import extract_feats.opensmile as of
import extract_feats.librosa as lf
from utils.opts import parse_opt
import models

def train(config) -> None:

    if config.feature_method == 'o':
        x_train, x_test, y_train, y_test = of.load_feature(config, train=True)

    elif config.feature_method == 'm':
        x_train, x_test, y_train, y_test = lf.load_feature(config, train=True)

    model = models.make(config=config, n_feats=x_train.shape[1])

    print('----- start training', config.model, '-----')
    if config.model in ['lstm', 'cnn1d', 'cnn2d']:
        y_train, y_val = to_categorical(y_train), to_categorical(y_test)
        model.train(
            x_train, y_train, x_test, y_val,
            batch_size = config.batch_size,
            n_epochs = config.epochs
        )

    else:
        model.train(x_train, y_train)

    print('----- training complete', config.model ,' -----')

    model.evaluate(x_test, y_test)
    model.save(config.checkpoint_path, config.checkpoint_name)

if __name__ == '__main__':
    config = parse_opt()
    train(config)