# 为什么是svd？？？怎么实现svd？？/
# 如果1280 + 720 = 2000， 如果直接将 1280*720的图片卷积映射到2000， 那不是简单的压缩吗？跟训练一个自编吗器有区别吗？
# 要贯彻svd fcn的思路， 必须得保证运算匹配关系啊， 我们建立的是 点到点的映射
# 因为自编吗网络两侧不能单单调整一侧，所以训练完毕后，用于压缩图片时结构、参数是确定的， 结果也是确定的，不能更新。
# 采用我们的结构，不仅少了一半网络需要训练， 还能够短时间内调整输出，这时只需要优化一个参数层， 而自编吗网络如果freze一些层也像
# 我们一样调整输出，是很难的，因为会影响其他图片的效果。而且想来也没我们快。最牛的是，我们这个结构可以同步地更新网络？？

# 读数据存储，正常读取
# 训练网络， 先不用非线性的激活函数
# batch 训练，一个batch 1280 *720
#
from numpy import *
import zlib
import os
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as Data
import sys
import time
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def yuv44p_import(filename, dims):
    fp0 = open(filename, 'rb')
    bytes_temp = fp0.read()
    blk_size = dims[0]*dims[1]*3
    numpix = len(bytes_temp) // blk_size
    fp0.close()

    fp = open(filename, 'rb')
    Y = []
    U = []
    V = []
    Yt = np.zeros((dims[0], dims[1]), np.uint8, 'c')
    Ut = np.zeros((dims[0], dims[1]), np.uint8, 'c')
    Vt = np.zeros((dims[0], dims[1]), np.uint8, 'c')
    for i in range(numpix):
        for m in range(dims[0]):
            for n in range(dims[1]):
                Yt[m, n] = ord(fp.read(1))
        for m in range(dims[0]):
            for n in range(dims[1]):
                Ut[m, n] = ord(fp.read(1))
        for m in range(dims[0]):
            for n in range(dims[1]):
                Vt[m, n] = ord(fp.read(1))
        Y = Y + [Yt]
        U = U + [Ut]
        V = V + [Vt]
    fp.close()
    # store the lsit in array
    yuv_list_to_array = np.array((Y, U, V))
    np.save("yuv_list_to_array.npy", yuv_list_to_array)
    return numpix


def prepare_taining_data(path, data, dims, data_type):
    # 将每张图片分解成1280*720个像素向量用于训练， 以后可以尝试在线制作， 因为每次也是1280*720个向量一个batch
    # 这里的每张图片是yuv三个通道中的一个
    d = dims[0]*dims[1]
    data_x = np.zeros((len(data[0])*d, 1, dims[1]+dims[0]), dtype=np.int8)
    data_y_y = np.zeros((len(data[0]) * dims[0] * dims[1], 1, 1), dtype=np.int8)
    # data_u_y = np.zeros((len(data[0]) * dims[0] * dims[1], 1, 1), dtype=np.int8)
    # data_v_y = np.zeros((len(data[0]) * dims[0] * dims[1], 1, 1), dtype=np.int8)
    pix = 0
    i, j = 0, 0
    counter = 0
    while pix < len(data[0]):
        data_x[counter, 1, i] = 1
        data_x[counter, 1, j + dims[0]] = 1
        data_y_y[counter, 1, 1] = data[0][pix][i, j]
        # data_u_y[counter, 1, 1] = data[1][pix][i, j]
        # data_v_y[counter, 1, 1] = data[2][pix][i, j]

        counter += 1
        if i == dims[0] - 1:
            i = 0
            pix += 1
        else:
            if j == dims[1] - 1:
                j = 0
                i += 1
            else:
                j += 1

    # data done
    np.save(path + data_type + "data_x", data_x)
    np.save(path + data_type + "data_u_y", data_y_y)
    # np.save(path + data_type + "data_u_y", data_u_y)
    # np.save(path + data_type + "data_v_y", data_v_y)
    return 0


def train(ckpt_path, data_type, data_x, batch_size=1, last_epoch=0, start_epoch=0, total_epoch=501, RESUME=False, dims=[720, 1280]):
    # 没办法存储所需的所有data_x, x and y 维度是否可以不同，x是重复的
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    size_dim = dims[0]+dims[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(size_dim, size_dim),
        torch.nn.Linear(size_dim, 2),
        # 预测的结果
        torch.nn.Linear(2, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
       # torch.nn.Sigmoid(),
    )
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000], gamma=0.1)
    loss_func = torch.nn.MSELoss()

    if RESUME:
        print("loading exist model")
        path_checkpoint = "D:/liu laing/zx_bisai/trained_models/checkpoint/"
        path_checkpoint = path_checkpoint + "ckpt_" + str(last_epoch) + ".pth"
        print("load", path_checkpoint)
        checkpoint = torch.load(path_checkpoint)

        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])
    else:
        print("train from random")
    print("strat training, start epoch", start_epoch)
    # lr_temp = 0.1
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr_temp
    accumulation_steps = 1280*720 // batch_size
    print("start clock")
    time_start = time.time()
    for epoch in range(start_epoch, total_epoch):
        # 每个epoch 过一遍图片
        #for pix in range(len(data_x)):
        for pix in range(len(data_x) - 5):
            data_pix = (np.array(data_x[pix]) + 200) / 400  # 720 * 1280
            j = 0
            loss_cpu = 0
            while j < 1280:
                xi = np.zeros((720, 1, dims[0] + dims[1]))
                for i in range(720):
                    xi[i, 0, i] = data_pix[i, j]
                    xi[i, 0, 720+j] = data_pix[i, j]
                yi = data_pix[:, j].reshape(720, 1, 1)
                xi = Variable(torch.from_numpy(xi)).to(torch.float32)
                yi = Variable(torch.from_numpy(yi)).to(torch.float32)
                xi, yi = xi.to(device), yi.to(device)
                pred = net(xi)
                loss = loss_func(pred, yi)
                loss_cpu += loss
                loss = loss / accumulation_steps

                loss.backward()
                j += 1
            time_temp = time.time()
            print("epcoch:", epoch, "pix",pix, "loss sum of 1280*720", loss_cpu)
            print("total time:", time_temp - time_start)
            # 一张图 1280*720 次，更新一次参数
            optimizer.step()
            optimizer.zero_grad()
            time.sleep(60)
        lr_schedule.step()
        # if epoch % 2000 == 0:# check the loss:
        #     pass
        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            'lr_schedule': lr_schedule.state_dict()
        }
        print(epoch)
        if not os.path.isdir(ckpt_path):
            os.mkdir(ckpt_path)
        torch.save(checkpoint, ckpt_path + "/" + data_type + "_5_6_%s.pth" % (str(epoch)))
    return epoch


def encode(data, checkpoint_file, store_file):
    data_x = data
    print("load", checkpoint_file)
    checkpoint = torch.load(checkpoint_file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    size_dim = 1280+720
    net = torch.nn.Sequential(
        torch.nn.Linear(size_dim, size_dim),
        torch.nn.Linear(size_dim, 2),
        # 预测的结果
        torch.nn.Linear(2, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    net.load_state_dict(checkpoint['net'])
    net.to(device)

    loss_func = torch.nn.MSELoss()

    from collections.abc import Iterable

    def set_freeze_by_idxs(model, idxs, freeze=True):
        if not isinstance(idxs, Iterable):
            idxs = [idxs]
        num_child = len(list(model.children()))
        idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
        for idx, child in enumerate(model.children()):
            if idx not in idxs:
                continue
            for param in child.parameters():
                param.requires_grad = not freeze

    def freeze_by_idxs(model, idxs):
        set_freeze_by_idxs(model, idxs, True)
    freeze_by_idxs(net, [1, 2, 3, 4, 5, 6, 7])
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.9)
    print("start clock")
    time_start = time.time()
    accumulation_steps = 1280 * 720
    max_epoch = 5
    # xi = np.random.rand(2000).reshape(1, -1)
    # xi = np.expand_dims(xi, 0)
    # xi = xi.repeat(720, axis=0)
    xi = np.zeros((720 * 1280, 1, size_dim))
    j = 0
    while j < 1280:
        for i in range(720):
            xi[720*j + i, 0, i] = 1
            xi[720*j + i, 0, 720 + j] = 1
        j += 1
    for pix in range(len(data_x)):
        for epoch in range(max_epoch):
            data_pix = np.array(data_x[pix])  # 720 * 1280
            loss_cpu = 0

            xi = Variable(torch.from_numpy(xi)).to(torch.float32)
            xi = xi.to(device)
            pred = net(xi)
            del xi
            yi = data_pix.reshape(720 * 1280, 1, 1) / 130
            yi = Variable(torch.from_numpy(yi)).to(torch.float32)
            yi = yi.to(device)
            loss = loss_func(pred, yi)
            del yi
            loss_cpu += loss
            # loss = loss / accumulation_steps
            loss.backward()
            time_temp = time.time()
            print("epcoch:", epoch, "pix", pix, "loss sum of 1280*720", loss_cpu)
            print("total time:", time_temp - time_start)
            # 一张图 1280*720 次，更新一次参数
            optimizer.step()
            optimizer.zero_grad()

    sys.exit(0)

    input_data = np.zeros((dims[0]+dims[1], dims[0]+dims[1]))
    for i in range(dims[0]+dims[1]):
        input_data[i, i] = 1


    decode_array = np.zeros((data_x.shape[0], dims[0]+dims[1])) # 每个dims[0]+dims[1] 向量存一张图片
    # 封存后面的网络，训练第一层， 达到阈值存储中间值

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0)
    loss_func = torch.nn.MSELoss()
    for epoch in range(100):
        x = torch.from_numpy(input_data)
        x = Variable(x).to(torch.float32)
        pred = net(x)
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()



    with torch.no_grad():
        for i in range(data_x.shape[0]):

            x = net[0](x)
            x = net[1](x)
            x = net[2](x)
            x = net[3](x)
            decode_array[i, :] = x.detach().numpy()
            print(x)
    while True:
        minv = min(decode_array[0, :])
        if int(minv) == minv:
            break
        else:
            decode_array = decode_array * 10
    decode_array = decode_array.reshape(1, -1)

    with open(store_file, 'wb') as f:
        sum_byte = b''
        for i in range(decode_array.shape[1]):
            sum_byte = sum_byte + decode_array[0, i].tobytes()
        sum_byte_com = zlib.compress(sum_byte, level=9)
        print(len(sum_byte_com))
        f.write(sum_byte_com)


def decode(data, file, decode_file, output_layer_dim, ckpt_file):
    # 读出byte， decomprees， 反量化， 过神经网络， 规范化为0-255， decompress， 重新存储
    print("load" + data)
    data_x = np.load(data)


    fp = open(file, 'rb')
    square_size = [72, 128]
    width = 1280
    height = 720
    sum_byte_com = fp.read()

    # decompress, 得到量化的网络输出
    sum_byte = zlib.decompress(sum_byte_com)

    number_length = len(sum_byte) // data_x.shape[0]
    # 把所有图片的各个纬度存成了一个向量,先把这个向量读出来再reshape， 完全按照逆行的格式
    decode_list = []
    in_byte_length = len(sum_byte) // (data_x.shape[0] * output_layer_dim)  # 每一个量化值的byte长度
    for i in range(data_x.shape[0] * output_layer_dim):
        in_byte = sum_byte[i, i + in_byte_length]
        decode_list.append(np.frombuffer(in_byte, dtype=np.int16))  # ???? 不太确定转换的对不对
    decode_array = np.array(decode_list)
    decode_array.reshape(data_x.shape[0], output_layer_dim)
    # 读取完毕， 现在反量化,
    max_pos = decode_array.max()
    while True:
        if max_pos < 1:
            break
        decode_array = decode_array / 10
    # 反量化完毕，过神经网络
    model = torch.load(ckpt_file)

    encode_x = np.zeros(data_x.size)
    with torch.no_grad():
        for i in range(data_x.shape[0]):
            x = torch.from_numpy(decode_array[i, :])
            x = Variable(x).to(torch.float32)

            x = model[4](x)
            x = model[5](x)
            x = model[6](x)
            # net output
            # 反量化
            encode_output = x.detach().numpy()  # (-1, 1) should be
            # 输出等同于data_x 的数据类型， data_x 在制作时已经量化到（-1， 1），我们期望的收敛值也相应在（-1， 1）
            encode_x[i, :] = encode_output
    # 针对data_x的生成过程反量化 （-1，1）to （-128，128）， tobyte， decompres， 分块还原
    encode_x_decom = np.zeros((data_x.shape[0], width * height * 3))  # 储存还原的分块读取的图像
    for i in range(data_x.shape[0]):
        encode_output = (encode_x[i, :] * 128).trunc().tobytes()  # other choice to integer?
        encode_output_decom = zlib.decompress(encode_output)
        encode_x_decom[i, :] = np.frombuffer(encode_output_decom, dtype=np.int8).reshape((1, -1))
        # encode_x_decom : yuv yuv yuv......, 分块读取
    # 反量化
    # web [72, 128]
    # excel [72, 128]
    # ppt [72, 128]
    # word [72, 128]
    # for simplify, we use the same strategy for all data set
    with open(decode_file, 'wb') as f:
        for pix in range(data_x.shape[0]):
            pix_data_temp = encode_x_decom[pix, :]
            pix_data_temp = pix_data_temp.tolist()
            y_width_eight_data = np.zeros((height, width))
            u_width_eight_data = np.zeros((height, width))
            v_width_eight_data = np.zeros((height, width))

            square_x = 0
            square_y = 0
            while square_x < square_size[0] and square_y < square_size[1]:
                for m in range(square_size[0]):
                    for n in range(square_size[1]):
                        y_width_eight_data[
                            square_x * square_size[0] + m, square_y * square_size[1] + n] = pix_data_temp.pop(0)
                        u_width_eight_data[
                            square_x * square_size[0] + m, square_y * square_size[1] + n] = pix_data_temp.pop(0)
                        v_width_eight_data[
                            square_x * square_size[0] + m, square_y * square_size[1] + n] = pix_data_temp.pop(0)
                if square_x == square_size[0] - 1:
                    square_y += 1
                    square_x = 0
                else:
                    square_x += 1
            yy = y_width_eight_data.reshape(1, -1)
            uu = u_width_eight_data.reshape(1, -1)
            vv = v_width_eight_data.reshape(1, -1)
            # 检查数据类型是不是int8， 如果是直接tobyte是可以的
            f.write(yy.tobytes())
            f.write(uu.tobytes())
            f.write(vv.tobytes())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 在一个函数内实现不同阶段的encode，decode


if __name__ == '__main__':
    setup_seed(0)
    height, width = 720, 1280
    stage = 'train'
    print(stage)
    data_path = "D:/liu laing/zx_bisai/DLFC_DATA/"
    yuv_path = 'D:/liu laing/zx_bisai/data/'
    test_amount_dict = {'excel_01': 137, 'ppt_02': 106, 'web_03': 120, 'word_04': 169}# ppt 106 web 120 # excel 137  #word  169

    data_type = "web_03"
    ckpt_path = "D:/liu laing/zx_bisai/trained_models/checkpoint_dlfc"

    if stage == 'train':
        # train model, store model
        filename = yuv_path + data_type + ".yuv"
        test_amount = test_amount_dict['web_03']

        # data_x, data_y,u,v_y 四个文件，存在一个即验证完毕
        # 为了节省内存，一律不在return数据， 而是从文件中读取

        if os.path.exists(data_path + data_type + "data_x" +".npy"):
            print("load data form ", data_path + data_type + "data_x" +".npy")
        else:
            print("no data set, prepare data set now")
            if os.path.exists(data_path + data_type + "yuv_list_to_array" +".npy"):
                print("load yuv_list_to_array")
            else:
                print(" no yuv_list_to_array")
                numpix = yuv44p_import(filename, (height, width))
            yuv_data = np.load(data_path + data_type + "yuv_list_to_array" + ".npy").astype(np.int8)
            yuv_data = yuv_data.tolist()

            #prepare_taining_data(data_path, yuv_data, (height, width), data_type)
        # data_x = np.load(data_path + data_type + "data_x" +".npy")
        # data_y_y = np.load(data_path + data_type + "data_y_y" + ".npy")
        # data_u_y = np.load(data_path + data_type + "data_u_y" + ".npy")
        # data_v_Y = np.load(data_path + data_type + "data_v_y" + ".npy")

        data_x = yuv_data[0]
        epoch = train(ckpt_path, data_type, data_x, batch_size=1)   # 先试一下y通道
    elif stage == 'encode':
        yuv_data = np.load(data_path + data_type + "yuv_list_to_array" + ".npy").astype(np.int8)
        yuv_data = yuv_data.tolist()
        data_x = yuv_data[0]
        epoch = 1
        checkpoint_file = ckpt_path + "/" + data_type + "_%s.pth" % (str(epoch))
        store_file = data_type + '.code'
        # web 250 , ppt 200  excel 150, # word 190
        #训练模型和encode过程使用的data不同，需要单独制作一次，此处同一批数据既用来训练，也是encode 的对象
        encode(data_x, checkpoint_file, store_file)
    else:
        print("decode .code files")
        file =  '.code'
        epoch = 40
        checkpoint_file = ckpt_path + "/" + + "_%s.pth" % (str(epoch))
        decode_file = "decode" + "web03.yuv"

        decode(file, decode_file, checkpoint_file)








    path = " D:/liu laing/zx_bisai/DLFC_DATA/"
    height = 720
    width = 1280
    filename = []
    data, numpix = yuv44p_import(filename[0], (height, width))
    prepare_taining_data(path, data, (height, width))
    device = torch.device('duda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100

    data_x = Variable(torch.from_numpy(data_x).to(torch.float32))

    torch_dataset = Data.TensorDataset(data_x, data_y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )


