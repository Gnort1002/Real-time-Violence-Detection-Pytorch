import matplotlib.pyplot as plt
import json

def visual(train_info_path):
    """Visualize the training process.

    Args:
        train_info_path(str): The path of the train information file.
    """
    f = open(train_info_path, 'r')
    info = json.load(f)

    train_losses = info['train_losses']
    train_scores = info['train_scores']
    test_losses = info['test_losses']
    test_scores = info['test_scores']

    print(train_losses)

    # Plot the training loss curve
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('./train_loss_curve.png')

    # Plot the training score curve
    plt.figure()
    plt.plot(train_scores)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training Score Curve')
    plt.savefig('./train_score_curve.png')

    # Plot the test loss curve
    plt.figure()
    plt.plot(test_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss Curve')
    plt.savefig('./test_loss_curve.png')

    # Plot the test score curve
    plt.figure()
    plt.plot(test_scores)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Test Score Curve')
    plt.savefig('./test_score_curve.png')


if __name__ == '__main__':
    visual('./train_info.json')
