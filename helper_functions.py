import itertools
from sklearn.metrics import confusion_matrix
import random
import matplotlib.pyplot as plt
import seaborn as sns



def plot_loss_accuracy_curves(history):
  """
  Returns separate loss/accuracy curves for training and validation metrics
  """
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"]))

  plt.plot(epochs, loss, "b.-", label="training loss")
  plt.plot(epochs, val_loss,  "r.-", label="validation loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.grid()
  plt.legend()

  plt.figure()
  plt.plot(epochs, accuracy, "b.-", label="training accuracy")
  plt.plot(epochs, val_accuracy,  "r.-", label="validation accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.grid()
  plt.legend()
  


def plot_decison_boundary(model, X, y):
  """
  Plots the decison boundary created by a model predicting on X
  """
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                       np.linspace(y_min, y_max, 100))
  x_in = np.c_[xx.ravel(), yy.ravel()]

  y_pred = model.predict(x_in)
  if len(y_pred[0]) > 1:
    print("multiclass classification")
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("binary classification")
    y_pred = np.round(y_pred).reshape(xx.shape)
  
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())

plot_decison_boundary(model, X, y)


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10,10), text_size=15):
  conf = confusion_matrix(y_true, y_pred)
  conf_norm = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis] # normalize confusion matrix
  c_classes = conf.shape[0]

  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(conf, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom()
  ax.yaxis.label.set_size(text_size+2)
  ax.xaxis.label.set_size(text_size+2)
  ax.title.set_size(text_size+5)
  thresh = (conf.max() + conf.min())/ 2.


  if classes:
    labels = classes
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
      plt.text(j, i, f"{conf[i,j]} ({conf_norm[i , j]*100:.1f}%)", horizontalalignment="center", color="white" if conf[i,j] > thresh else "black", size=text_size)
  else:
    labels = np.arange(conf.shape[0])
    group_names = ['True Positives', 'False Negatives','False Positives','True Negatives']
    k = 0
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
      plt.text(j, i, f"{conf[i,j]} ({conf_norm[i , j]*100:.1f}%) {group_names[k]}", horizontalalignment="center", color="white" if conf[i,j] > thresh else "black", size=text_size)
      k+=1
  ax.set(title="Confusion Matrix", xlabel="Predicted Label", ylabel="True Label", xticks=np.arange(c_classes), yticks=np.arange(c_classes), xticklabels=labels, yticklabels=labels)


def plot_random_image(model, images, true_labels, class_names):
  """
  Picks random image, plots it, and labels it with prediction and truth label
  """
  i = random.randint(0, len(images))

  target_image = images[i]
  pred_probs = model.predict(target_image.reshape(1, 28, 28))
  pred_label = class_names[pred_probs.argmax()]
  true_label = class_names[true_labels[i]]

  plt.imshow(target_image, cmap=plt.cm.binary)

  if pred_label == true_label:
    color = "green"
  else:
    color = "red"
  
  plt.xlabel("Pred: {}  {:2.0f}% (True: {})".format(pred_label, 100*tf.reduce_max(pred_probs), true_label), color=color)

  
  
def display_WordCloud(tokens):
  """Creates a Word Cloud for @tokens, with all lowercase letters, with the size of each word based on frequency """
  comment_words = ''
  stopwords = set(STOPWORDS)

  for i in range(len(tokens)):
    tokens[i] = tokens[i].lower()
      
  comment_words += " ".join(tokens)+" "

  wordcloud = WordCloud(width = 800, height = 800,
                  background_color ='white',
                  stopwords = stopwords,
                  min_font_size = 10).generate(comment_words)
  plt.figure(figsize = (8, 8), facecolor = None)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout(pad = 0)
  
  plt.show()
  
  
  profile = ProfileReport(df)
profile.to_file(output_file = f"{FIGURES_DIR}/pima_diabetes/diabetes.html")
profile.to_notebook_iframe()
