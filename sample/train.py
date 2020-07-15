import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from tqdm.autonotebook import tqdm

from sample import CodeDataset
from sample.seq2seq import Seq2Seq, Decoder, Encoder

torch.manual_seed(0)


# This will be used to automatically pad all batch items to the same length
def pad_collate(batch):
    data = [item[0] for item in batch]
    data = pad_sequence(data)
    targets = [item[1] for item in batch]
    targets = pad_sequence(targets)
    return [data, targets]


if __name__ == "__main__":
        # Load the data and split randomly into training and val subsets
    ds = CodeDataset()
    tr, va = random_split(ds, [len(ds) - len(ds) // 3, len(ds) // 3])
    trainloader = DataLoader(tr, batch_size=1024, shuffle=True, collate_fn=pad_collate)
    valloader = DataLoader(va, batch_size=1024, shuffle=False, collate_fn=pad_collate)

    INPUT_DIM = len(ds.morsebet)
    OUTPUT_DIM = len(ds.alphabet)
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM)
    model = Seq2Seq(enc, dec).cuda()

    crit = nn.CrossEntropyLoss(ignore_index=ds.PAD_IDX)
    opt = optim.Adam(model.parameters())

    epoch = 10
    e_plot = np.zeros((epoch, 1))

    for e in range(epoch):
        model.train()
        plot_loss = 0
        count = 0
        with tqdm(total=len(trainloader), desc='train') as t:
            epoch_loss = 0
            for i, (x, y) in enumerate(trainloader):
                x = x.cuda()
                y = y.cuda()

                opt.zero_grad()
                pred = model(x, y, padding_idx=ds.PAD_IDX)

                pred_dim = pred.shape[-1]
                pred = pred[1:].view(-1, pred_dim)
                y = y[1:].view(-1)

                loss = crit(pred, y)
                loss.backward()
                opt.step()

                epoch_loss = (epoch_loss * i + loss.item()) / (i + 1)
                plot_loss += loss.item()
                count += 1

                t.set_postfix(loss='{:05.3f}'.format(epoch_loss))
                t.update()

        e_plot[e] = plot_loss / count

        model.eval()
        with tqdm(total=len(valloader), desc='val') as t:
            with torch.no_grad():
                epoch_loss = 0
                for i, (x, y) in enumerate(valloader):
                    x = x.cuda()
                    y = y.cuda()

                    pred = model(x, y, teacher_forcing_ratio=0, padding_idx=ds.PAD_IDX)

                    pred_dim = pred.shape[-1]
                    pred = pred[1:].view(-1, pred_dim)
                    y = y[1:].view(-1)

                    loss = crit(pred, y)
                    epoch_loss = (epoch_loss * i + loss.item()) / (i + 1)

                    t.set_postfix(loss='{:05.3f}'.format(epoch_loss))
                    t.update()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(e_plot)
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.grid(True)
    plt.show()


    def decode(code):
        out = ''
        i = 0
        cur = []
        for chunk in code.split(' '):
            cur.append(chunk)
            if i % 5 == 0:
                cur = ' '.join(cur)
                print(cur)
                num = ds.encode_morse('^ ' + cur + ' $').unsqueeze(1)
                pred = model(num.cuda(), maxlen=2)
                pred = pred[1:].view(-1, pred_dim).argmax(-1)
                out += ds.decode_alpha(pred.cpu())[::-1]
                cur = []
            i += 1

        return out


    print(decode('.- -. ... .-- . .-. / - .... . / ..-. --- .-.. .-.. --- .-- .. -. --.'))
