import torch
from torch import nn
import struct
import argparse
from config import get_config
from Learner import face_learner

def main():
    parser = argparse.ArgumentParser(description='for generate weights file.')
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)

    args = parser.parse_args()

    conf = get_config(False)
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'ir_se50.pth', True, True)
    print('learner loaded')

    net = learner.model
    print('cuda device count: ', torch.cuda.device_count())
    net = net.to('cuda:0')
    net.eval()

    tmp = torch.ones(1,3,112, 112).to('cuda:0')
    out = net(tmp)
    print('arcface out:', out)

    f = open("arcface-r50.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == '__main__':
    main()