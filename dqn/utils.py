import os

def makedir_if_there_is_no(path):
  if not os.path.exists(path):
    os.makedirs(path)
    print("****Directory {} was made".format(path))
