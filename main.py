
import tensorflow as tf
from config import *
from network.eval import Learning

def main(argv=None):
    Learning()


if __name__ == '__main__':
    tf.app.run()
