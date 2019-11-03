import glob
for fname in glob.glob('NHWC/val2017/*'):
    base_name = fname.split('/')[-1].split('.')[0]
    print(fname, base_name)
    data = open(fname,'rb').read()[128:]
    assert len(data) == 300*300*3
    open(base_name+'.bin', 'wb').write(data)
