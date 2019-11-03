import glob

x = len('ILSVRC2012_val_00000001')
for fname in glob.glob('*.npy'):
    print(fname[:x])
    data = open(fname, 'rb').read()[-224*224*3:]
    open(fname[:x]+'.bin','wb').write(data)

