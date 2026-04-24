import matplotlib.pyplot as plt
from IPython import display

plt.ion() # Interactive mode on

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Evolutionary Training Progress')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.plot(scores, label='Max Fitness')
    plt.plot(mean_scores, label='Mean Fitness')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(round(scores[-1], 2)))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)