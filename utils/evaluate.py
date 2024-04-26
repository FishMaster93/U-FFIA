from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from utils.pytorch_utils import forward_audio, forward_video, forward_av


class Evaluator(object):
    def __init__(self, model):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model

    def evaluate_audio(self, data_loader):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict, 
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward_audio(
            model=self.model,
            generator=data_loader,
            return_target=True)

        clipwise_output = output_dict['clipwise_output']  # (audios_num, classes_num)
        target = output_dict['target']  # (audios_num, classes_num)

        # AP
        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None)

        auc = metrics.roc_auc_score(target, clipwise_output, average=None)  # AUC under ROC

        target_acc = np.argmax(target, axis=1)
        clipwise_output_acc = np.argmax(clipwise_output, axis=1)
        acc = accuracy_score(target_acc, clipwise_output_acc)

        cm = confusion_matrix(target_acc, clipwise_output_acc)
        # cm_display = ConfusionMatrixDisplay(cm).plot()

        message = classification_report(target_acc, clipwise_output_acc)
        message = '\n' + message
        statistics = {'average_precision': average_precision, 'accuracy': acc, 'auc': auc, 'message': message,
                      'confu_matrix': cm}

        return statistics

    def evaluate_video(self, data_loader):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict,
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward_video(
            model=self.model,
            generator=data_loader,
            return_target=True)

        clipwise_output = output_dict['clipwise_output']  # (audios_num, classes_num)
        target = output_dict['target']  # (audios_num, classes_num)

        # AP
        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None)

        auc = metrics.roc_auc_score(target, clipwise_output, average=None)  # AUC under ROC

        target_acc = np.argmax(target, axis=1)
        clipwise_output_acc = np.argmax(clipwise_output, axis=1)
        acc = accuracy_score(target_acc, clipwise_output_acc)

        cm = confusion_matrix(target_acc, clipwise_output_acc)

        message = classification_report(target_acc, clipwise_output_acc)
        message = '\n' + message
        statistics = {'average_precision': average_precision, 'accuracy': acc, 'auc': auc, 'message': message,
                      'confu_matrix': cm}

        return statistics

    def evaluate_av(self, data_loader):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict,
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward_av(
            model=self.model,
            generator=data_loader,
            return_target=True)

        clipwise_output = output_dict['clipwise_output']  # (audios_num, classes_num)
        target = output_dict['target']  # (audios_num, classes_num)

        # AP
        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None)

        auc = metrics.roc_auc_score(target, clipwise_output, average=None)  # AUC under ROC

        target_acc = np.argmax(target, axis=1)
        clipwise_output_acc = np.argmax(clipwise_output, axis=1)
        acc = accuracy_score(target_acc, clipwise_output_acc)

        cm = confusion_matrix(target_acc, clipwise_output_acc)
        # cm_display = ConfusionMatrixDisplay(cm).plot()

        message = classification_report(target_acc, clipwise_output_acc)
        message = '\n' + message
        statistics = {'average_precision': average_precision, 'accuracy': acc, 'auc': auc, 'message': message,
                      'confu_matrix': cm}

        return statistics