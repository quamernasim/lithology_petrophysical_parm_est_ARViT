from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from utils.misc import fit_classical_ml_and_get_metrics

def train_classical_ml(config, x_train, y_train, x_val, y_val):

    trainer_config = config['model']
    classical_ml_config = trainer_config['classical_ml']

    classifier = RandomForestClassifier(n_estimators=classical_ml_config['random_forest']['n_estimators'], 
                                        random_state=config['random_state'], n_jobs = -1, verbose = 1)
    classifier, train_accuracy, validation_accuracy = fit_classical_ml_and_get_metrics(classifier, x_train, y_train, x_val, y_val)

    classifier = MLPClassifier(hidden_layer_sizes=classical_ml_config['ann']['hidden_layers'], 
                                        random_state=config['random_state'], verbose = 1)
    classifier, train_accuracy, validation_accuracy = fit_classical_ml_and_get_metrics(classifier, x_train, y_train, x_val, y_val)