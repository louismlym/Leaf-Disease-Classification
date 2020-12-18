import os

dic = {}
total = 0
for dirc in os.listdir('../data/train'):
  dirPath = os.path.join('../data/train', dirc)
  if os.path.isdir(dirPath):
    count = 0
    for fi in os.listdir(dirPath):
      fiPath = os.path.join(dirPath, fi)
      if os.path.isfile(fiPath):
        count = count + 1
    dic[dirc] = count
    total += count

temp = 0
for k, v in sorted(dic.items()):
  temp += v
  print("| {} | {} |".format(i, k, temp))

print("total: {} images".format(total))